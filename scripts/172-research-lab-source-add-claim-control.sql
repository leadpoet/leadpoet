-- Linearize SOURCE_ADD work claims, pause control, and restart quiescence.
--
-- Migration 145 serializes initial miner admission and pause changes with the
-- source-add-control advisory lock.  Queue claims must take that same lock
-- before reading the paused state so a successful pause plus quiescence
-- snapshot proves that no later claim can create a leased work item.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Replacing the claim authority is safe only at a paused, fully drained
-- SOURCE_ADD handoff.  Count every leased row regardless of lease expiry: an
-- expired lease may still represent a provider request which has started and
-- whose outcome is not yet known.  The table locks reject an N-1 transaction
-- which is still reading control or mutating work instead of straddling the
-- function replacement.
DO $quiet_pause$
BEGIN
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF COALESCE((
        SELECT control.paused
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
    ), FALSE) IS NOT TRUE THEN
        RAISE EXCEPTION 'SOURCE_ADD must be paused before claim-control migration';
    END IF;
    LOCK TABLE public.research_lab_source_add_work_items
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items work
        WHERE work.work_status = 'leased'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD work is leased during claim-control migration';
    END IF;
END;
$quiet_pause$;

DO $preflight$
BEGIN
    IF pg_catalog.to_regprocedure('extensions.digest(bytea,text)') IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD claim-control authority requires extensions.digest';
    END IF;
END;
$preflight$;

ALTER TABLE public.research_lab_source_add_control
    ADD COLUMN IF NOT EXISTS restart_guard_commitment TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS restart_guard_owner_commitment TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS restart_guard_generation BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS restart_guard_expires_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS restart_guard_acquired_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS restart_guard_actor_ref TEXT NOT NULL DEFAULT '';

ALTER TABLE public.research_lab_source_add_control
    DROP CONSTRAINT IF EXISTS
        research_lab_source_add_control_restart_guard_check;
ALTER TABLE public.research_lab_source_add_control
    ADD CONSTRAINT research_lab_source_add_control_restart_guard_check
    CHECK (
        restart_guard_generation >= 0
        AND (
            (
                restart_guard_commitment = ''
                AND restart_guard_owner_commitment = ''
                AND restart_guard_expires_at IS NULL
                AND restart_guard_acquired_at IS NULL
                AND restart_guard_actor_ref = ''
            ) OR (
                restart_guard_commitment ~ '^sha256:[0-9a-f]{64}$'
                AND restart_guard_owner_commitment
                    ~ '^sha256:[0-9a-f]{64}$'
                AND restart_guard_generation > 0
                AND restart_guard_expires_at IS NOT NULL
                AND restart_guard_acquired_at IS NOT NULL
                AND restart_guard_actor_ref <> ''
            )
        )
    );

CREATE OR REPLACE FUNCTION public.research_lab_source_add_claim_work(
    p_worker_id TEXT,
    p_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_row public.research_lab_source_add_work_items%ROWTYPE;
    v_token UUID := gen_random_uuid();
BEGIN
    IF p_worker_id = '' OR p_lease_seconds < 30 OR p_lease_seconds > 900 THEN
        RAISE EXCEPTION 'SOURCE_ADD work lease input is invalid';
    END IF;

    -- This lock is deliberately acquired before the paused read.  Pause,
    -- admission, claim, and restart quiescence therefore have one total order.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    IF COALESCE((
        SELECT paused
        FROM public.research_lab_source_add_control
        WHERE singleton
    ), TRUE) THEN
        RETURN jsonb_build_object('status', 'paused');
    END IF;
    SELECT w.* INTO v_row
    FROM public.research_lab_source_add_work_items w
    WHERE (
        (w.work_status IN ('queued', 'retry_wait') AND w.available_at <= NOW())
        OR (w.work_status = 'leased' AND w.lease_expires_at <= NOW())
    )
      AND (
        w.work_kind NOT IN ('functional_probe', 'provisioning_smoke')
        OR COALESCE(w.job_doc->>'host_hash', '') = ''
        OR NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_work_items active
            WHERE active.work_id <> w.work_id
              AND active.work_status = 'leased'
              AND active.lease_expires_at > NOW()
              AND active.job_doc->>'host_hash' = w.job_doc->>'host_hash'
        )
      )
    ORDER BY w.priority ASC, w.available_at ASC, w.created_at ASC, w.work_id ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'empty'); END IF;

    IF v_row.work_kind IN ('functional_probe', 'provisioning_smoke')
       AND COALESCE(v_row.job_doc->>'host_hash', '') <> '' THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(
                'source-add-host:' || (v_row.job_doc->>'host_hash'), 0
            )
        );
        IF EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_work_items active
            WHERE active.work_id <> v_row.work_id
              AND active.work_status = 'leased'
              AND active.lease_expires_at > NOW()
              AND active.job_doc->>'host_hash' = v_row.job_doc->>'host_hash'
        ) THEN
            RETURN jsonb_build_object('status', 'host_busy');
        END IF;
    END IF;

    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'leased',
        -- A crashed worker's expired lease reuses the same deterministic V2
        -- operation/attempt. Explicit retry_wait transitions advance attempts.
        attempt_count = CASE
            WHEN v_row.work_status = 'leased' THEN attempt_count
            ELSE LEAST(attempt_count + 1, 20)
        END,
        lease_token = v_token,
        leased_by = p_worker_id,
        lease_expires_at = NOW() + make_interval(secs => p_lease_seconds),
        job_doc = CASE
            WHEN v_row.work_status = 'leased'
                 AND v_row.job_doc->>'provider_execution_state' = 'started'
            THEN v_row.job_doc || jsonb_build_object(
                'provider_execution_recovery', 'uncertain_after_lease_expiry'
            )
            ELSE v_row.job_doc - 'provider_execution_recovery'
        END,
        updated_at = NOW()
    WHERE work_id = v_row.work_id
    RETURNING * INTO v_row;
    RETURN jsonb_build_object('status', 'claimed', 'work', to_jsonb(v_row));
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_restart_guard_state_v1()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_paused BOOLEAN;
    v_guard_commitment TEXT;
    v_owner_commitment TEXT;
    v_guard_generation BIGINT;
    v_guard_expires_at TIMESTAMPTZ;
    v_guard_active BOOLEAN;
    v_owner_generation_commitment TEXT := '';
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    SELECT
        control.paused,
        control.restart_guard_commitment,
        control.restart_guard_owner_commitment,
        control.restart_guard_generation,
        control.restart_guard_expires_at
    INTO
        v_paused,
        v_guard_commitment,
        v_owner_commitment,
        v_guard_generation,
        v_guard_expires_at
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    v_guard_active := v_guard_commitment <> ''
        AND v_guard_expires_at > pg_catalog.clock_timestamp();
    IF v_owner_commitment <> '' THEN
        v_owner_generation_commitment := 'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    v_owner_commitment || ':' || v_guard_generation::TEXT,
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_guard_state.v1',
        'paused', v_paused,
        'guard_active', v_guard_active,
        'guard_commitment', v_guard_commitment,
        'owner_commitment', v_owner_commitment,
        'guard_generation', v_guard_generation,
        'owner_generation_commitment', v_owner_generation_commitment,
        'guard_expires_at', v_guard_expires_at
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_acquire_restart_guard_v1(
    p_guard_id TEXT,
    p_owner_id TEXT,
    p_expected_generation BIGINT,
    p_lease_seconds INTEGER,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_guard_commitment TEXT;
    v_owner_commitment TEXT;
    v_existing_commitment TEXT;
    v_existing_owner_commitment TEXT;
    v_existing_generation BIGINT;
    v_existing_expires_at TIMESTAMPTZ;
    v_next_generation BIGINT;
    v_owner_generation_commitment TEXT;
    v_now TIMESTAMPTZ;
    v_expires_at TIMESTAMPTZ;
BEGIN
    IF COALESCE(p_guard_id, '')
          !~ '^source_add_restart_guard:[0-9a-f]{64}$'
       OR COALESCE(p_owner_id, '')
          !~ '^source_add_restart_owner:[0-9a-f]{64}$'
       OR COALESCE(p_expected_generation, -1) < 0
       OR COALESCE(p_lease_seconds, 0) NOT BETWEEN 60 AND 14400
       OR COALESCE(btrim(p_actor_ref), '') = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard input is invalid';
    END IF;
    v_guard_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_guard_id, 'UTF8'),
            'sha256'
        ),
        'hex'
    );
    v_owner_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_owner_id, 'UTF8'),
            'sha256'
        ),
        'hex'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    v_now := pg_catalog.clock_timestamp();
    SELECT
        control.restart_guard_commitment,
        control.restart_guard_owner_commitment,
        control.restart_guard_generation,
        control.restart_guard_expires_at
    INTO
        v_existing_commitment,
        v_existing_owner_commitment,
        v_existing_generation,
        v_existing_expires_at
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    IF v_existing_generation <> p_expected_generation THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard generation differs';
    END IF;
    IF v_existing_commitment <> ''
       AND v_existing_expires_at > v_now
       AND v_existing_commitment <> v_guard_commitment THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard is already active';
    END IF;

    IF v_existing_commitment = v_guard_commitment
       AND v_existing_owner_commitment = v_owner_commitment
       AND v_existing_expires_at > v_now THEN
        -- An exact active owner/generation replay is a bounded renewal. It
        -- changes only the lease expiry, preserving the legacy five-field
        -- control commitment already bound into the restart proof.
        v_next_generation := v_existing_generation;
        v_expires_at := GREATEST(
            v_existing_expires_at,
            v_now + pg_catalog.make_interval(secs => p_lease_seconds)
        );
        UPDATE public.research_lab_source_add_control
        SET restart_guard_expires_at = v_expires_at
        WHERE singleton;
    ELSE
        IF v_existing_generation = 9223372036854775807 THEN
            RAISE EXCEPTION 'SOURCE_ADD restart guard generation is exhausted';
        END IF;
        v_next_generation := v_existing_generation + 1;
        v_expires_at := v_now
            + pg_catalog.make_interval(secs => p_lease_seconds);
        UPDATE public.research_lab_source_add_control
        SET paused = TRUE,
            reason = 'canonical_restart_guard',
            actor_ref = left(p_actor_ref, 200),
            updated_at = NOW(),
            restart_guard_commitment = v_guard_commitment,
            restart_guard_owner_commitment = v_owner_commitment,
            restart_guard_generation = v_next_generation,
            restart_guard_expires_at = v_expires_at,
            restart_guard_acquired_at = v_now,
            restart_guard_actor_ref = left(p_actor_ref, 200)
        WHERE singleton;
    END IF;
    v_owner_generation_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                v_owner_commitment || ':' || v_next_generation::TEXT,
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_guard.v1',
        'paused', TRUE,
        'guard_active', TRUE,
        'guard_commitment', v_guard_commitment,
        'owner_commitment', v_owner_commitment,
        'guard_generation', v_next_generation,
        'owner_generation_commitment', v_owner_generation_commitment,
        'guard_expires_at', v_expires_at
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_set_paused(
    p_paused BOOLEAN,
    p_reason TEXT,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_guard_commitment TEXT;
BEGIN
    IF COALESCE(btrim(p_reason), '') = ''
       OR COALESCE(btrim(p_actor_ref), '') = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD pause reason and actor are required';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    SELECT control.restart_guard_commitment
    INTO v_guard_commitment
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    -- An expired guard remains an explicit recovery obligation. It may be
    -- safely reacquired while paused and then released by exact id, but must
    -- never turn into an implicit resume merely because wall time advanced.
    IF p_paused IS NOT TRUE AND v_guard_commitment <> '' THEN
        RAISE EXCEPTION
            'SOURCE_ADD restart guard must be explicitly reacquired and released before resume';
    END IF;
    UPDATE public.research_lab_source_add_control
    SET paused = p_paused,
        reason = left(p_reason, 500),
        actor_ref = left(p_actor_ref, 200),
        updated_at = NOW()
    WHERE singleton;
    RETURN (
        SELECT to_jsonb(control)
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_release_restart_guard_v1(
    p_guard_id TEXT,
    p_owner_id TEXT,
    p_guard_generation BIGINT,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_guard_commitment TEXT;
    v_owner_commitment TEXT;
    v_owner_generation_commitment TEXT;
    v_existing_commitment TEXT;
    v_existing_owner_commitment TEXT;
    v_existing_generation BIGINT;
BEGIN
    IF COALESCE(p_guard_id, '')
          !~ '^source_add_restart_guard:[0-9a-f]{64}$'
       OR COALESCE(p_owner_id, '')
          !~ '^source_add_restart_owner:[0-9a-f]{64}$'
       OR COALESCE(p_guard_generation, 0) <= 0
       OR COALESCE(btrim(p_actor_ref), '') = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard release input is invalid';
    END IF;
    v_guard_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_guard_id, 'UTF8'),
            'sha256'
        ),
        'hex'
    );
    v_owner_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_owner_id, 'UTF8'),
            'sha256'
        ),
        'hex'
    );
    v_owner_generation_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                v_owner_commitment || ':' || p_guard_generation::TEXT,
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    SELECT
        control.restart_guard_commitment,
        control.restart_guard_owner_commitment,
        control.restart_guard_generation
    INTO
        v_existing_commitment,
        v_existing_owner_commitment,
        v_existing_generation
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    IF v_existing_commitment = ''
       OR v_existing_commitment <> v_guard_commitment
       OR v_existing_owner_commitment <> v_owner_commitment
       OR v_existing_generation <> p_guard_generation THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard owner or generation does not match';
    END IF;
    UPDATE public.research_lab_source_add_control
    SET paused = TRUE,
        reason = 'canonical_restart_guard_released_paused',
        actor_ref = left(p_actor_ref, 200),
        updated_at = NOW(),
        restart_guard_commitment = '',
        restart_guard_owner_commitment = '',
        restart_guard_expires_at = NULL,
        restart_guard_acquired_at = NULL,
        restart_guard_actor_ref = ''
    WHERE singleton;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_guard_release.v1',
        'released', TRUE,
        'paused', TRUE,
        'guard_active', FALSE,
        'guard_generation', p_guard_generation,
        'owner_generation_commitment', v_owner_generation_commitment
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_restart_quiescence_v1(
    p_guard_id TEXT,
    p_owner_id TEXT,
    p_guard_generation BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_control_present BOOLEAN := FALSE;
    v_paused BOOLEAN := FALSE;
    v_expected_guard_commitment TEXT := '';
    v_expected_owner_commitment TEXT := '';
    v_guard_commitment TEXT := '';
    v_owner_commitment TEXT := '';
    v_guard_generation BIGINT := 0;
    v_owner_generation_commitment TEXT := '';
    v_guard_expires_at TIMESTAMPTZ;
    v_guard_active BOOLEAN := FALSE;
    v_guard_matches BOOLEAN := FALSE;
    v_owner_matches BOOLEAN := FALSE;
    v_generation_matches BOOLEAN := FALSE;
    v_leased_work_count INTEGER := 0;
BEGIN
    IF COALESCE(p_guard_id, '')
          !~ '^source_add_restart_guard:[0-9a-f]{64}$'
       OR COALESCE(p_owner_id, '')
          !~ '^source_add_restart_owner:[0-9a-f]{64}$'
       OR COALESCE(p_guard_generation, 0) <= 0 THEN
        RAISE EXCEPTION 'SOURCE_ADD restart quiescence guard input is invalid';
    END IF;
    v_expected_guard_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_guard_id, 'UTF8'),
            'sha256'
        ),
        'hex'
    );
    v_expected_owner_commitment := 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(p_owner_id, 'UTF8'),
            'sha256'
        ),
        'hex'
    );
    -- A guard-bound paused=true/leased_work_count=0 snapshot is serialized
    -- against every candidate claim and resume. Claims retain the transaction
    -- lock through their lease write, and resume remains forbidden until the
    -- exact guard is explicitly released after verified restart completion.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    SELECT
        control.paused,
        control.restart_guard_commitment,
        control.restart_guard_owner_commitment,
        control.restart_guard_generation,
        control.restart_guard_expires_at
    INTO
        v_paused,
        v_guard_commitment,
        v_owner_commitment,
        v_guard_generation,
        v_guard_expires_at
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    v_control_present := FOUND;
    v_guard_active := v_control_present
        AND v_guard_commitment <> ''
        AND v_guard_expires_at > pg_catalog.clock_timestamp();
    v_guard_matches := v_control_present
        AND v_guard_commitment = v_expected_guard_commitment;
    v_owner_matches := v_control_present
        AND v_owner_commitment = v_expected_owner_commitment;
    v_generation_matches := v_control_present
        AND v_guard_generation = p_guard_generation;
    IF v_owner_commitment <> '' THEN
        v_owner_generation_commitment := 'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    v_owner_commitment || ':' || v_guard_generation::TEXT,
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        );
    END IF;

    SELECT COUNT(*)::INTEGER
    INTO v_leased_work_count
    FROM public.research_lab_source_add_work_items work
    WHERE work.work_status = 'leased';

    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_quiescence.v1',
        'paused', v_control_present AND v_paused,
        'guard_active', v_guard_active,
        'guard_matches', v_guard_matches,
        'owner_matches', v_owner_matches,
        'generation_matches', v_generation_matches,
        'guard_commitment', CASE
            WHEN v_control_present THEN v_guard_commitment ELSE ''
        END,
        'owner_commitment', CASE
            WHEN v_control_present THEN v_owner_commitment ELSE ''
        END,
        'guard_generation', CASE
            WHEN v_control_present THEN v_guard_generation ELSE 0
        END,
        'owner_generation_commitment', v_owner_generation_commitment,
        'guard_expires_at', v_guard_expires_at,
        'leased_work_count', v_leased_work_count,
        'quiescent', (
            v_control_present
            AND v_paused
            AND v_guard_active
            AND v_guard_matches
            AND v_owner_matches
            AND v_generation_matches
            AND v_leased_work_count = 0
        )
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_claim_control_contract_v1()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_service_role_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role'
    ) INTO v_service_role_exists;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_claim_control_contract.v1',
        'control_lock', 'source-add-control',
        'pause_rpc', 'research_lab_source_add_set_paused',
        'pause_signature', 'boolean,text,text',
        'claim_rpc', 'research_lab_source_add_claim_work',
        'claim_signature', 'text,integer',
        'acquire_guard_rpc',
            'research_lab_source_add_acquire_restart_guard_v1',
        'acquire_guard_signature', 'text,text,bigint,integer,text',
        'guard_state_rpc',
            'research_lab_source_add_restart_guard_state_v1',
        'guard_state_signature', '',
        'release_guard_rpc',
            'research_lab_source_add_release_restart_guard_v1',
        'release_guard_signature', 'text,text,bigint,text',
        'guard_state_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'paused', 'guard_active',
            'guard_commitment', 'owner_commitment', 'guard_generation',
            'owner_generation_commitment', 'guard_expires_at'
        ),
        'acquire_guard_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'paused', 'guard_active',
            'guard_commitment', 'owner_commitment', 'guard_generation',
            'owner_generation_commitment', 'guard_expires_at'
        ),
        'release_guard_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'released', 'paused', 'guard_active',
            'guard_generation', 'owner_generation_commitment'
        ),
        'guard_id_format',
            '^source_add_restart_guard:[0-9a-f]{64}$',
        'guard_commitment', 'sha256_utf8_guard_id',
        'owner_id_format',
            '^source_add_restart_owner:[0-9a-f]{64}$',
        'owner_commitment', 'sha256_utf8_owner_id',
        'owner_generation_commitment',
            'sha256_utf8_owner_commitment_colon_decimal_generation',
        'guard_lease_min_seconds', 60,
        'guard_lease_max_seconds', 14400,
        'active_guard_replay_extends_lease', TRUE,
        'acquire_compare_and_swap', 'expected_generation',
        'different_owner_takeover_increments_generation', TRUE,
        'expired_reacquire_increments_generation', TRUE,
        'generation_retained_after_release', TRUE,
        'resume_requires_guard_clear', TRUE,
        'expired_guard_recovery',
            'explicit_reacquire_then_exact_release',
        'release_keeps_paused', TRUE,
        'restart_quiescence_rpc',
            'research_lab_source_add_restart_quiescence_v1',
        'restart_quiescence_signature', 'text,text,bigint',
        'restart_quiescence_schema_version',
            'leadpoet.source_add_restart_quiescence.v1',
        'restart_quiescence_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'paused', 'guard_active', 'guard_matches',
            'owner_matches', 'generation_matches',
            'guard_commitment', 'owner_commitment', 'guard_generation',
            'owner_generation_commitment', 'guard_expires_at',
            'leased_work_count', 'quiescent'
        ),
        'lock_before_paused_read', TRUE,
        'leased_scope', 'all_leased_regardless_of_expiry',
        'migration_requires_paused', TRUE,
        'migration_requires_zero_leased', TRUE,
        'function_authority_sha256', (
            SELECT 'sha256:' || pg_catalog.encode(
                extensions.digest(
                    pg_catalog.convert_to(
                        COALESCE(
                            pg_catalog.jsonb_object_agg(
                                authority.name,
                                pg_catalog.jsonb_build_object(
                                    'body', proc.prosrc,
                                    'security_definer', proc.prosecdef,
                                    'configuration', pg_catalog.to_jsonb(
                                        proc.proconfig
                                    ),
                                    'identity_arguments',
                                        pg_catalog.pg_get_function_identity_arguments(
                                            proc.oid
                                        ),
                                    'argument_names', pg_catalog.to_jsonb(
                                        proc.proargnames
                                    ),
                                    'language', language.lanname,
                                    'volatility', proc.provolatile,
                                    'parallel', proc.proparallel,
                                    'kind', proc.prokind,
                                    'return_type', proc.prorettype::REGTYPE::TEXT
                                )
                            ),
                            '{}'::JSONB
                        )::TEXT,
                        'UTF8'
                    ),
                    'sha256'
                ),
                'hex'
            )
            FROM (
                VALUES
                    (
                        'admission_guard',
                        'public.enforce_research_lab_source_add_admission_control()'
                    ),
                    (
                        'acquire_restart_guard_v1',
                        'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)'
                    ),
                    (
                        'claim_work',
                        'public.research_lab_source_add_claim_work(text,integer)'
                    ),
                    (
                        'contract_v1',
                        'public.research_lab_source_add_claim_control_contract_v1()'
                    ),
                    (
                        'pause',
                        'public.research_lab_source_add_set_paused(boolean,text,text)'
                    ),
                    (
                        'release_restart_guard_v1',
                        'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)'
                    ),
                    (
                        'restart_guard_state_v1',
                        'public.research_lab_source_add_restart_guard_state_v1()'
                    ),
                    (
                        'restart_quiescence_v1',
                        'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)'
                    )
            ) AS authority(name, signature)
            LEFT JOIN pg_catalog.pg_proc proc
              ON proc.oid = pg_catalog.to_regprocedure(authority.signature)
            LEFT JOIN pg_catalog.pg_language language
              ON language.oid = proc.prolang
        ),
        'functions', pg_catalog.jsonb_build_object(
            'admission_guard', pg_catalog.to_regprocedure(
                'public.enforce_research_lab_source_add_admission_control()'
            ) IS NOT NULL,
            'acquire_restart_guard_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)'
            ) IS NOT NULL,
            'claim_work', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_claim_work(text,integer)'
            ) IS NOT NULL,
            'pause', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_set_paused(boolean,text,text)'
            ) IS NOT NULL,
            'release_restart_guard_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)'
            ) IS NOT NULL,
            'restart_guard_state_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_restart_guard_state_v1()'
            ) IS NOT NULL,
            'restart_quiescence_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)'
            ) IS NOT NULL
        ),
        'permissions', pg_catalog.jsonb_build_object(
            'service_role_exists', v_service_role_exists,
            'acquire_guard_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'claim_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_claim_work(text,integer)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'pause_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_set_paused(boolean,text,text)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'quiescence_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'release_guard_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'guard_state_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_restart_guard_state_v1()',
                    'EXECUTE'
                ) ELSE FALSE END,
            'contract_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_claim_control_contract_v1()',
                    'EXECUTE'
                ) ELSE FALSE END,
            'anon_callable',
                pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_claim_work(text,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_set_paused(boolean,text,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_restart_guard_state_v1()',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_claim_control_contract_v1()',
                    'EXECUTE'
                ),
            'authenticated_callable',
                pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_claim_work(text,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_set_paused(boolean,text,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_restart_guard_state_v1()',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_claim_control_contract_v1()',
                    'EXECUTE'
                )
        )
    );
END;
$function$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_claim_work(TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_acquire_restart_guard_v1(
    TEXT, TEXT, BIGINT, INTEGER, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_set_paused(
    BOOLEAN, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_release_restart_guard_v1(
    TEXT, TEXT, BIGINT, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_restart_guard_state_v1()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_restart_quiescence_v1(
    TEXT, TEXT, BIGINT
)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_claim_control_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_claim_work(TEXT, INTEGER)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_acquire_restart_guard_v1(
    TEXT, TEXT, BIGINT, INTEGER, TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_set_paused(
    BOOLEAN, TEXT, TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_release_restart_guard_v1(
    TEXT, TEXT, BIGINT, TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_restart_guard_state_v1()
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_restart_quiescence_v1(
    TEXT, TEXT, BIGINT
)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_claim_control_contract_v1()
    TO service_role;

COMMENT ON FUNCTION public.research_lab_source_add_claim_work(TEXT, INTEGER) IS
    'Claims one SOURCE_ADD work item only after serializing with the durable pause-control advisory lock.';
COMMENT ON FUNCTION public.research_lab_source_add_acquire_restart_guard_v1(
    TEXT, TEXT, BIGINT, INTEGER, TEXT
) IS
    'Acquires one bounded identity-bound canonical restart guard and atomically pauses SOURCE_ADD.';
COMMENT ON FUNCTION public.research_lab_source_add_release_restart_guard_v1(
    TEXT, TEXT, BIGINT, TEXT
) IS
    'Releases only an exact SOURCE_ADD restart guard identity and deliberately leaves SOURCE_ADD paused.';
COMMENT ON FUNCTION public.research_lab_source_add_restart_guard_state_v1() IS
    'Returns the exact monotonic SOURCE_ADD restart guard generation for compare-and-swap acquisition.';
COMMENT ON FUNCTION public.research_lab_source_add_restart_quiescence_v1(
    TEXT, TEXT, BIGINT
) IS
    'Returns a guard-bound pause-and-all-leased-work snapshot serialized against SOURCE_ADD claims and resume.';
COMMENT ON FUNCTION public.research_lab_source_add_claim_control_contract_v1() IS
    'Read-only exact function-authority, lock-order, drain-scope, and ACL contract for SOURCE_ADD claims and restart quiescence.';

NOTIFY pgrst, 'reload schema';

COMMIT;
