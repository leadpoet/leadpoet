-- Preserve SOURCE_ADD operator state across a successful canonical restart.
--
-- The restart guard still pauses and drains SOURCE_ADD before cutover.  The
-- guard now captures the prior pause state durably and restores it atomically
-- with exact owner/generation release.  Failed restarts remain paused.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $quiet_pause$
BEGIN
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF COALESCE((
        SELECT control.paused
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
    ), FALSE) IS NOT TRUE THEN
        RAISE EXCEPTION 'SOURCE_ADD must be paused before restart-state migration';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
          AND control.restart_guard_commitment <> ''
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard is active during restart-state migration';
    END IF;
    LOCK TABLE public.research_lab_source_add_work_items
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items work
        WHERE work.work_status = 'leased'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD work is leased during restart-state migration';
    END IF;
END;
$quiet_pause$;

ALTER TABLE public.research_lab_source_add_control
    ADD COLUMN IF NOT EXISTS restart_guard_restore_paused BOOLEAN;

CREATE OR REPLACE FUNCTION public.enforce_source_add_restart_restore_pause_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
BEGIN
    IF NEW.restart_guard_commitment = '' THEN
        NEW.restart_guard_restore_paused := NULL;
    ELSIF OLD.restart_guard_commitment = '' THEN
        -- First acquisition captures the operator state immediately before
        -- the same UPDATE forces the guarded pause.
        NEW.restart_guard_restore_paused := OLD.paused;
    ELSIF NEW.restart_guard_commitment = OLD.restart_guard_commitment
       AND NEW.restart_guard_owner_commitment
           = OLD.restart_guard_owner_commitment
       AND NEW.restart_guard_generation = OLD.restart_guard_generation THEN
        -- An explicit operator pause while the guard is held wins over an
        -- earlier active-state snapshot. Exact lease renewal changes neither
        -- reason nor actor and therefore preserves the snapshot.
        IF NEW.paused IS TRUE
           AND (
               NEW.reason IS DISTINCT FROM OLD.reason
               OR NEW.actor_ref IS DISTINCT FROM OLD.actor_ref
           ) THEN
            NEW.restart_guard_restore_paused := TRUE;
        END IF;
    ELSIF NEW.restart_guard_restore_paused IS NULL THEN
        -- A legacy/expired guard has no proved active-state snapshot. A new
        -- owner may recover it, but release remains conservatively paused.
        NEW.restart_guard_restore_paused := COALESCE(
            OLD.restart_guard_restore_paused,
            TRUE
        );
    END IF;
    RETURN NEW;
END;
$function$;

DROP TRIGGER IF EXISTS trg_source_add_restart_restore_pause_v2
    ON public.research_lab_source_add_control;
CREATE TRIGGER trg_source_add_restart_restore_pause_v2
BEFORE UPDATE ON public.research_lab_source_add_control
FOR EACH ROW
EXECUTE FUNCTION public.enforce_source_add_restart_restore_pause_v2();

ALTER TABLE public.research_lab_source_add_control
    DROP CONSTRAINT IF EXISTS
        research_lab_source_add_control_restart_restore_check;
ALTER TABLE public.research_lab_source_add_control
    ADD CONSTRAINT research_lab_source_add_control_restart_restore_check
    CHECK (
        (
            restart_guard_commitment = ''
            AND restart_guard_restore_paused IS NULL
        ) OR (
            restart_guard_commitment <> ''
            AND restart_guard_restore_paused IS NOT NULL
        )
    );

-- Defense in depth: new SOURCE_ADD event rows cannot persist credential-shaped
-- values such as BuiltWith's KEY= query parameter. NOT VALID avoids making
-- historical-row validation a deployment prerequisite while still enforcing
-- the check for every new or updated row.
ALTER TABLE public.research_lab_source_add_submissions
    DROP CONSTRAINT IF EXISTS
        research_lab_source_add_submission_no_credential_material_v2;
ALTER TABLE public.research_lab_source_add_submissions
    ADD CONSTRAINT research_lab_source_add_submission_no_credential_material_v2
    CHECK (
        submission_doc::TEXT !~* (
            '(^|[^[:alnum:]_])'
            || '(key|api[[:space:]_-]*key|access[[:space:]_-]*(key|token)|'
            || 'client[[:space:]_-]*secret|refresh[[:space:]_-]*token|'
            || 'private[[:space:]_-]*key|secret([[:space:]_-]*key)?|'
            || 'subscription[[:space:]_-]*key|token|password|credentials?)'
            || '["'']?[[:space:]]*(=|:|%3d)[[:space:]"'']*'
            || '[^[:space:]"'',;}]{8,}'
        )
        AND submission_doc::TEXT !~* (
            '(authorization|proxy-authorization)["'']?[[:space:]]*:'
            || '[[:space:]"'']*'
            || '(bearer|basic|api([[:space:]_-]*key)?)'
            || '[[:space:]]+[^[:space:]"'',;}]{8,}'
        )
        AND submission_doc::TEXT !~* (
            '["'']('
            || '[[:alnum:]_-]*(api|access|auth|client|private|provider|refresh|secret|subscription)'
            || '[[:alnum:]_-]*(key|token|secret|password|credentials?|auth|authorization)'
            || '|x[[:alnum:]_-]*(key|token|secret|auth|authorization)'
            || ')'
            || '["''][[:space:]]*:[[:space:]"'']*'
            || '[^[:space:]"'',;}]{8,}'
        )
    ) NOT VALID;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_restart_guard_state_v2()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_state JSONB;
    v_restore_paused BOOLEAN;
BEGIN
    -- v1 takes the shared transaction advisory lock before reading control.
    v_state := public.research_lab_source_add_restart_guard_state_v1();
    SELECT control.restart_guard_restore_paused
    INTO v_restore_paused
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    RETURN (v_state - 'schema_version') || pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_guard_state.v2',
        'restore_paused', v_restore_paused
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_acquire_restart_guard_v2(
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
    v_guard JSONB;
    v_restore_paused BOOLEAN;
BEGIN
    -- The trigger captures/preserves restore state inside v1's locked UPDATE.
    v_guard := public.research_lab_source_add_acquire_restart_guard_v1(
        p_guard_id,
        p_owner_id,
        p_expected_generation,
        p_lease_seconds,
        p_actor_ref
    );
    SELECT control.restart_guard_restore_paused
    INTO v_restore_paused
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND OR v_restore_paused IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD restart restore state is unavailable';
    END IF;
    RETURN (v_guard - 'schema_version') || pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_guard.v2',
        'restore_paused', v_restore_paused
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_release_restart_guard_v2(
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
    v_existing_reason TEXT;
    v_existing_actor_ref TEXT;
    v_restore_paused BOOLEAN;
    v_final_paused BOOLEAN;
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
        control.restart_guard_generation,
        control.restart_guard_restore_paused,
        control.reason,
        control.actor_ref
    INTO
        v_existing_commitment,
        v_existing_owner_commitment,
        v_existing_generation,
        v_restore_paused,
        v_existing_reason,
        v_existing_actor_ref
    FROM public.research_lab_source_add_control control
    WHERE control.singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    IF v_existing_commitment = ''
       OR v_existing_commitment <> v_guard_commitment
       OR v_existing_owner_commitment <> v_owner_commitment
       OR v_existing_generation <> p_guard_generation
       OR v_restore_paused IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD restart guard owner or generation does not match';
    END IF;
    -- An operator pause which changed reason/actor while the exact guard was
    -- held always wins. Resume remains forbidden until guard release.
    v_final_paused := v_restore_paused
        OR v_existing_reason <> 'canonical_restart_guard'
        OR v_existing_actor_ref <> left(p_actor_ref, 200);
    UPDATE public.research_lab_source_add_control
    SET paused = v_final_paused,
        reason = CASE
            WHEN v_final_paused
                THEN 'canonical_restart_guard_restored_paused'
            ELSE 'canonical_restart_guard_restored_active'
        END,
        actor_ref = left(p_actor_ref, 200),
        updated_at = NOW(),
        restart_guard_commitment = '',
        restart_guard_owner_commitment = '',
        restart_guard_expires_at = NULL,
        restart_guard_acquired_at = NULL,
        restart_guard_actor_ref = '',
        restart_guard_restore_paused = NULL
    WHERE singleton;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_restart_guard_release.v2',
        'released', TRUE,
        'paused', v_final_paused,
        'guard_active', FALSE,
        'guard_generation', p_guard_generation,
        'owner_generation_commitment', v_owner_generation_commitment,
        'restored_pre_restart_state', TRUE
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_claim_control_contract_v2()
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
        'schema_version', 'leadpoet.source_add_claim_control_contract.v2',
        'control_lock', 'source-add-control',
        'pause_rpc', 'research_lab_source_add_set_paused',
        'pause_signature', 'boolean,text,text',
        'claim_rpc', 'research_lab_source_add_claim_work',
        'claim_signature', 'text,integer',
        'acquire_guard_rpc',
            'research_lab_source_add_acquire_restart_guard_v2',
        'acquire_guard_signature', 'text,text,bigint,integer,text',
        'guard_state_rpc',
            'research_lab_source_add_restart_guard_state_v2',
        'guard_state_signature', '',
        'release_guard_rpc',
            'research_lab_source_add_release_restart_guard_v2',
        'release_guard_signature', 'text,text,bigint,text',
        'restart_quiescence_rpc',
            'research_lab_source_add_restart_quiescence_v1',
        'restart_quiescence_signature', 'text,text,bigint',
        'guard_state_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'paused', 'guard_active',
            'guard_commitment', 'owner_commitment', 'guard_generation',
            'owner_generation_commitment', 'guard_expires_at',
            'restore_paused'
        ),
        'acquire_guard_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'paused', 'guard_active',
            'guard_commitment', 'owner_commitment', 'guard_generation',
            'owner_generation_commitment', 'guard_expires_at',
            'restore_paused'
        ),
        'release_guard_result_fields', pg_catalog.jsonb_build_array(
            'schema_version', 'released', 'paused', 'guard_active',
            'guard_generation', 'owner_generation_commitment',
            'restored_pre_restart_state'
        ),
        'restore_state_column', 'restart_guard_restore_paused',
        'acquire_captures_pre_restart_paused', TRUE,
        'renewal_preserves_restore_state', TRUE,
        'expired_takeover_preserves_restore_state', TRUE,
        'operator_pause_wins', TRUE,
        'release_restores_pre_restart_state', TRUE,
        'failed_restart_keeps_paused', TRUE,
        'rollback_v1_contract_schema_version',
            'leadpoet.source_add_claim_control_contract.v1',
        'rollback_v1_contract_sha256', 'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    public.research_lab_source_add_claim_control_contract_v1()::TEXT,
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        ),
        'migration_requires_paused', TRUE,
        'migration_requires_zero_leased', TRUE,
        'migration_requires_guard_clear', TRUE,
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
                                    'configuration', pg_catalog.to_jsonb(proc.proconfig),
                                    'identity_arguments',
                                        pg_catalog.pg_get_function_identity_arguments(proc.oid),
                                    'argument_names', pg_catalog.to_jsonb(proc.proargnames),
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
                    ('admission_guard',
                     'public.enforce_research_lab_source_add_admission_control()'),
                    ('acquire_restart_guard_v1',
                     'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)'),
                    ('acquire_restart_guard_v2',
                     'public.research_lab_source_add_acquire_restart_guard_v2(text,text,bigint,integer,text)'),
                    ('claim_work',
                     'public.research_lab_source_add_claim_work(text,integer)'),
                    ('contract_v1',
                     'public.research_lab_source_add_claim_control_contract_v1()'),
                    ('contract_v2',
                     'public.research_lab_source_add_claim_control_contract_v2()'),
                    ('pause',
                     'public.research_lab_source_add_set_paused(boolean,text,text)'),
                    ('release_restart_guard_v1',
                     'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)'),
                    ('release_restart_guard_v2',
                     'public.research_lab_source_add_release_restart_guard_v2(text,text,bigint,text)'),
                    ('restart_guard_state_v1',
                     'public.research_lab_source_add_restart_guard_state_v1()'),
                    ('restart_guard_state_v2',
                     'public.research_lab_source_add_restart_guard_state_v2()'),
                    ('restart_quiescence_v1',
                     'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)'),
                    ('restore_trigger_v2',
                     'public.enforce_source_add_restart_restore_pause_v2()')
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
            'acquire_restart_guard_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_acquire_restart_guard_v2(text,text,bigint,integer,text)'
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
            'release_restart_guard_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_release_restart_guard_v2(text,text,bigint,text)'
            ) IS NOT NULL,
            'restart_guard_state_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_restart_guard_state_v1()'
            ) IS NOT NULL,
            'restart_guard_state_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_restart_guard_state_v2()'
            ) IS NOT NULL,
            'restart_quiescence_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)'
            ) IS NOT NULL,
            'restore_trigger_v2', pg_catalog.to_regprocedure(
                'public.enforce_source_add_restart_restore_pause_v2()'
            ) IS NOT NULL
        ),
        'permissions', pg_catalog.jsonb_build_object(
            'service_role_exists', v_service_role_exists,
            'service_role_callable', v_service_role_exists
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_acquire_restart_guard_v2(text,text,bigint,integer,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_release_restart_guard_v2(text,text,bigint,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_restart_guard_state_v2()',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_claim_control_contract_v1()',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_release_restart_guard_v1(text,text,bigint,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_restart_guard_state_v1()',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_claim_work(text,integer)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_set_paused(boolean,text,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_restart_quiescence_v1(text,text,bigint)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_claim_control_contract_v2()',
                    'EXECUTE'
                ),
            'anon_callable',
                pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_acquire_restart_guard_v2(text,text,bigint,integer,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_release_restart_guard_v2(text,text,bigint,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_restart_guard_state_v2()',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_claim_control_contract_v1()',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)',
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
                    'public.research_lab_source_add_claim_control_contract_v2()',
                    'EXECUTE'
                ),
            'authenticated_callable',
                pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_acquire_restart_guard_v2(text,text,bigint,integer,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_release_restart_guard_v2(text,text,bigint,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_restart_guard_state_v2()',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_claim_control_contract_v1()',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_acquire_restart_guard_v1(text,text,bigint,integer,text)',
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
                    'public.research_lab_source_add_claim_control_contract_v2()',
                    'EXECUTE'
                )
        )
    );
END;
$function$;

REVOKE ALL ON FUNCTION public.enforce_source_add_restart_restore_pause_v2()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_acquire_restart_guard_v2(
    TEXT, TEXT, BIGINT, INTEGER, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_release_restart_guard_v2(
    TEXT, TEXT, BIGINT, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_restart_guard_state_v2()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_claim_control_contract_v2()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_acquire_restart_guard_v2(
    TEXT, TEXT, BIGINT, INTEGER, TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_release_restart_guard_v2(
    TEXT, TEXT, BIGINT, TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_restart_guard_state_v2()
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_claim_control_contract_v2()
    TO service_role;

COMMENT ON COLUMN public.research_lab_source_add_control.restart_guard_restore_paused IS
    'Guard-bound operator pause state restored atomically after successful canonical restart verification.';
COMMENT ON FUNCTION public.research_lab_source_add_release_restart_guard_v2(
    TEXT, TEXT, BIGINT, TEXT
) IS
    'Releases an exact SOURCE_ADD restart guard and atomically restores its captured pre-restart pause state.';
COMMENT ON CONSTRAINT research_lab_source_add_submission_no_credential_material_v2
    ON public.research_lab_source_add_submissions IS
    'Rejects new SOURCE_ADD event documents containing credential-shaped assignments or authorization values.';

NOTIFY pgrst, 'reload schema';

COMMIT;
