-- Classify durable SOURCE_ADD duplicates before the per-hotkey route cooldown.
--
-- Migration 168 owns exact-host uniqueness and remains unchanged.  This
-- wrapper adds one admission ordering guarantee: requests sharing a durable
-- submission, work, source-identity, documentation-identity, legacy-identity,
-- catalog identity, or exact provider host are classified as duplicates before
-- a distinct-source request can be rejected by the route cooldown.  The
-- cooldown is derived from durable provenance work, so gateway restarts and
-- concurrent gateway processes cannot reset or race it.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Replacing v2 with the rolling-release compatibility wrapper is safe only at
-- the same paused, drained SOURCE_ADD handoff required by migrations 167 and
-- 168.  NOWAIT makes a request which is still writing any affected durable
-- state abort this migration for a clean retry instead of straddling the
-- function-authority change.
DO $quiet_pause$
BEGIN
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF NOT COALESCE((
        SELECT paused
        FROM public.research_lab_source_add_control
        WHERE singleton
    ), FALSE) THEN
        RAISE EXCEPTION 'SOURCE_ADD must be paused before duplicate-privacy migration';
    END IF;
    LOCK TABLE
        public.research_lab_source_add_work_items,
        public.research_lab_source_add_submissions,
        public.research_lab_source_add_identity_events,
        public.research_lab_source_add_functional_probe_attempts,
        public.research_lab_source_catalog,
        public.research_lab_source_add_provisioning_events,
        public.research_lab_source_add_reward_intents,
        public.research_lab_source_add_reward_obligations,
        public.research_lab_source_add_provider_origin_events
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items
        WHERE work_status = 'leased'
          AND work_kind IN (
              'provenance', 'functional_probe',
              'provisioning_smoke', 'leg1_reward'
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD work is leased during duplicate-privacy migration';
    END IF;
END;
$quiet_pause$;

DO $preflight$
BEGIN
    IF pg_catalog.to_regprocedure('extensions.digest(bytea,text)') IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD duplicate-privacy authority requires extensions.digest';
    END IF;
END;
$preflight$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit_v3(
    p_record_doc JSONB,
    p_identity_hash TEXT,
    p_documentation_identity_hash TEXT,
    p_legacy_identity_hash TEXT,
    p_provider_origin_hash TEXT,
    p_work_id TEXT,
    p_max_open INTEGER,
    p_max_day INTEGER,
    p_max_30d INTEGER,
    p_cooldown_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_submission_id TEXT := p_record_doc->>'submission_id';
    v_adapter_id TEXT := p_record_doc->>'adapter_id';
    v_miner_hotkey TEXT := p_record_doc->>'miner_hotkey';
    v_api_base_url TEXT := p_record_doc #>> '{source_metadata,api_base_url}';
    v_origin_host TEXT;
    v_record_doc JSONB;
    v_result JSONB;
    v_lock_key TEXT;
    v_latest_submission_at TIMESTAMPTZ;
    v_now TIMESTAMPTZ;
    v_seq INTEGER;
    v_wait_seconds INTEGER;
BEGIN
    v_origin_host := public.research_lab_source_add_provider_origin_host_v1(
        v_api_base_url
    );
    IF COALESCE(v_submission_id, '')
          !~ '^source_add_submission:[0-9a-f]{16}$'
       OR COALESCE(v_adapter_id, '') = ''
       OR COALESCE(v_miner_hotkey, '') = ''
       OR COALESCE(p_identity_hash, '') !~ '^sha256:[0-9a-f]{64}$'
       OR (COALESCE(p_documentation_identity_hash, '') <> ''
           AND p_documentation_identity_hash !~ '^sha256:[0-9a-f]{64}$')
       OR (COALESCE(p_legacy_identity_hash, '') <> ''
           AND p_legacy_identity_hash !~ '^sha256:[0-9a-f]{64}$')
       OR COALESCE(p_provider_origin_hash, '') !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(p_work_id, '') !~ '^source_add_work:[0-9a-f]{16}$'
       OR COALESCE(p_max_open, 0) < 1
       OR COALESCE(p_max_day, 0) < 1
       OR COALESCE(p_max_30d, 0) < 1
       OR COALESCE(p_cooldown_seconds, 0) NOT BETWEEN 1 AND 3600
       OR COALESCE(jsonb_typeof(p_record_doc), '') <> 'object'
       OR COALESCE(p_record_doc->'credential_envelope', '{}'::JSONB)
          <> '{}'::JSONB
       OR COALESCE(p_record_doc->'manifest'->>'credential_policy', '')
          <> 'no_credentials'
       OR COALESCE(p_record_doc->'manifest'->>'credential_ref', '') <> ''
       OR p_record_doc::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
       OR COALESCE(v_origin_host, '') = ''
       OR p_provider_origin_hash
          <> public.research_lab_source_add_provider_origin_hash_v1(
                 v_api_base_url
             )
       OR COALESCE(p_record_doc->>'provider_origin_host', '')
          NOT IN ('', v_origin_host)
       OR COALESCE(p_record_doc->>'provider_origin_hash', '')
          NOT IN ('', p_provider_origin_hash) THEN
        RAISE EXCEPTION 'SOURCE_ADD duplicate-private admission input is invalid';
    END IF;

    -- Every key which can make this request already-known, plus the hotkey
    -- whose limits govern a new request, is locked in one deterministic
    -- order.  The existing v2 function takes the overlapping origin/identity
    -- locks before its hotkey lock, so the first two ranks preserve that order
    -- during a rolling exact-release transition.
    FOR v_lock_key IN
        SELECT lock_key
        FROM (
            SELECT DISTINCT 0 AS lock_rank, item AS lock_key
            FROM unnest(ARRAY[
                'source-add-provider-origin:' || p_provider_origin_hash,
                'source-add-identity:' || p_identity_hash,
                CASE WHEN COALESCE(p_documentation_identity_hash, '') = ''
                     THEN NULL ELSE
                        'source-add-identity:' || p_documentation_identity_hash
                END,
                CASE WHEN COALESCE(p_legacy_identity_hash, '') = ''
                     THEN NULL ELSE
                        'source-add-identity:' || p_legacy_identity_hash
                END
            ]) item
            WHERE item IS NOT NULL
            UNION ALL
            SELECT 1, 'source-add-hotkey:' || v_miner_hotkey
            UNION ALL
            SELECT 2, 'source-add-submission:' || v_submission_id
            UNION ALL
            SELECT 2, 'source-add-work:' || p_work_id
        ) ordered_locks
        ORDER BY lock_rank, lock_key
    LOOP
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(v_lock_key, 0)
        );
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submission_current current
        WHERE current.submission_id = v_submission_id
          AND current.stage NOT IN (
              'rejected', 'rejected_precheck', 'functional_probe_failed'
          )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items work
        WHERE work.work_id = p_work_id
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_provider_origin_current origin
        WHERE origin.provider_origin_hash = p_provider_origin_hash
          AND origin.reservation_status = 'reserved'
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_identity_current identity
        WHERE identity.reservation_status = 'reserved'
          AND identity.source_identity_hash IN (
              p_identity_hash,
              NULLIF(p_documentation_identity_hash, ''),
              NULLIF(p_legacy_identity_hash, '')
          )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_catalog catalog
        WHERE catalog.source_identity_hash IN (
            p_identity_hash,
            NULLIF(p_documentation_identity_hash, ''),
            NULLIF(p_legacy_identity_hash, '')
        )
    ) THEN
        RETURN jsonb_build_object('status', 'duplicate');
    END IF;

    -- Distinct identities from one miner are already serialized by the hotkey
    -- lock above.  The durable work timestamp is written by admission in this
    -- transaction, making cooldown enforcement survive restarts and cover
    -- concurrent gateway processes.
    SELECT MAX(work.created_at), pg_catalog.clock_timestamp()
    INTO v_latest_submission_at, v_now
    FROM public.research_lab_source_add_work_items work
    JOIN public.research_lab_source_add_submission_current current
      ON current.submission_id = work.submission_id
    WHERE current.miner_hotkey = v_miner_hotkey
      AND work.work_kind = 'provenance'
      AND work.job_doc->>'admission_kind' = 'miner_submission';

    IF v_latest_submission_at IS NOT NULL
       AND v_latest_submission_at
           > v_now - (p_cooldown_seconds * INTERVAL '1 second') THEN
        v_wait_seconds := GREATEST(
            1,
            CEIL(EXTRACT(EPOCH FROM (
                v_latest_submission_at
                + (p_cooldown_seconds * INTERVAL '1 second')
                - v_now
            )))::INTEGER
        );
        RETURN jsonb_build_object(
            'status', 'route_cooldown',
            'cooldown_seconds', p_cooldown_seconds,
            'wait_seconds', v_wait_seconds
        );
    END IF;

    -- Reserve the provider origin under the rank-zero advisory lock, then use
    -- the unchanged v1 admission authority for limits and durable creation.
    -- This inlines the small v2 overlay so v2 can become a compatibility
    -- wrapper without creating a v2 <-> v3 recursion.
    SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
    FROM public.research_lab_source_add_provider_origin_events
    WHERE provider_origin_hash = p_provider_origin_hash;
    INSERT INTO public.research_lab_source_add_provider_origin_events (
        origin_version, provider_origin_hash, submission_id, adapter_id,
        miner_hotkey, reservation_status, seq, reason
    ) VALUES (
        'v1', p_provider_origin_hash, v_submission_id, v_adapter_id,
        v_miner_hotkey, 'reserved', v_seq, 'atomic_admission_v3'
    );
    v_record_doc := p_record_doc || jsonb_build_object(
        'provider_origin_host', v_origin_host,
        'provider_origin_hash', p_provider_origin_hash
    );
    v_result := public.research_lab_source_add_admit(
        v_record_doc,
        p_identity_hash,
        p_documentation_identity_hash,
        p_legacy_identity_hash,
        p_work_id,
        p_max_open,
        p_max_day,
        p_max_30d
    );
    IF COALESCE(v_result->>'status', '') <> 'admitted' THEN
        INSERT INTO public.research_lab_source_add_provider_origin_events (
            origin_version, provider_origin_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES (
            'v1', p_provider_origin_hash, v_submission_id, v_adapter_id,
            v_miner_hotkey, 'released', v_seq + 1,
            'admission_v3_not_admitted'
        );
    END IF;
    RETURN v_result;
END;
$function$;

-- N-1 gateways call v2.  Once this migration commits, those calls must use
-- the identical lock/classification authority as candidate v3 calls so a
-- rolling overlap cannot turn a duplicate into a uniqueness exception/503.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit_v2(
    p_record_doc JSONB,
    p_identity_hash TEXT,
    p_documentation_identity_hash TEXT,
    p_legacy_identity_hash TEXT,
    p_provider_origin_hash TEXT,
    p_work_id TEXT,
    p_max_open INTEGER,
    p_max_day INTEGER,
    p_max_30d INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
BEGIN
    RETURN public.research_lab_source_add_admit_v3(
        p_record_doc,
        p_identity_hash,
        p_documentation_identity_hash,
        p_legacy_identity_hash,
        p_provider_origin_hash,
        p_work_id,
        p_max_open,
        p_max_day,
        p_max_30d,
        20
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_duplicate_privacy_contract_v1()
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
        'schema_version', 'leadpoet.source_add_duplicate_privacy_contract.v1',
        'admission_rpc', 'research_lab_source_add_admit_v3',
        'admission_signature',
            'jsonb,text,text,text,text,text,integer,integer,integer,integer',
        'compatibility_rpc', 'research_lab_source_add_admit_v2',
        'compatibility_signature',
            'jsonb,text,text,text,text,text,integer,integer,integer',
        'compatibility_cooldown_seconds', 20,
        'cooldown_parameter_min_seconds', 1,
        'cooldown_parameter_max_seconds', 3600,
        'cooldown_clock', 'clock_timestamp_after_advisory_locks',
        'cooldown_source', 'durable_miner_provenance_work',
        'duplicate_precedes_cooldown', TRUE,
        'lock_order', pg_catalog.jsonb_build_array(
            'provider_origin_or_identity',
            'hotkey',
            'submission_or_work'
        ),
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
                        'admit_v1',
                        'public.research_lab_source_add_admit(jsonb,text,text,text,text,integer,integer,integer)'
                    ),
                    (
                        'admit_v2_compatibility',
                        'public.research_lab_source_add_admit_v2(jsonb,text,text,text,text,text,integer,integer,integer)'
                    ),
                    (
                        'admit_v3',
                        'public.research_lab_source_add_admit_v3(jsonb,text,text,text,text,text,integer,integer,integer,integer)'
                    ),
                    (
                        'contract_v1',
                        'public.research_lab_source_add_duplicate_privacy_contract_v1()'
                    ),
                    (
                        'provider_origin_hash_v1',
                        'public.research_lab_source_add_provider_origin_hash_v1(text)'
                    ),
                    (
                        'provider_origin_host_v1',
                        'public.research_lab_source_add_provider_origin_host_v1(text)'
                    )
            ) AS authority(name, signature)
            LEFT JOIN pg_catalog.pg_proc proc
              ON proc.oid = pg_catalog.to_regprocedure(authority.signature)
            LEFT JOIN pg_catalog.pg_language language
              ON language.oid = proc.prolang
        ),
        'functions', pg_catalog.jsonb_build_object(
            'admit_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_admit(jsonb,text,text,text,text,integer,integer,integer)'
            ) IS NOT NULL,
            'admit_v2_compatibility', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_admit_v2(jsonb,text,text,text,text,text,integer,integer,integer)'
            ) IS NOT NULL,
            'admit_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_admit_v3(jsonb,text,text,text,text,text,integer,integer,integer,integer)'
            ) IS NOT NULL,
            'provider_origin_hash_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_provider_origin_hash_v1(text)'
            ) IS NOT NULL,
            'provider_origin_host_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_provider_origin_host_v1(text)'
            ) IS NOT NULL
        ),
        'permissions', pg_catalog.jsonb_build_object(
            'service_role_exists', v_service_role_exists,
            'v3_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_admit_v3(jsonb,text,text,text,text,text,integer,integer,integer,integer)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'v2_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_admit_v2(jsonb,text,text,text,text,text,integer,integer,integer)',
                    'EXECUTE'
                ) ELSE FALSE END,
            'contract_service_role_callable', CASE
                WHEN v_service_role_exists THEN pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_duplicate_privacy_contract_v1()',
                    'EXECUTE'
                ) ELSE FALSE END,
            'anon_callable',
                pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_admit_v3(jsonb,text,text,text,text,text,integer,integer,integer,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_admit_v2(jsonb,text,text,text,text,text,integer,integer,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'anon',
                    'public.research_lab_source_add_duplicate_privacy_contract_v1()',
                    'EXECUTE'
                ),
            'authenticated_callable',
                pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_admit_v3(jsonb,text,text,text,text,text,integer,integer,integer,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_admit_v2(jsonb,text,text,text,text,text,integer,integer,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_duplicate_privacy_contract_v1()',
                    'EXECUTE'
                )
        )
    );
END;
$function$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_admit_v3(
    JSONB, TEXT, TEXT, TEXT, TEXT, TEXT,
    INTEGER, INTEGER, INTEGER, INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_admit_v2(
    JSONB, TEXT, TEXT, TEXT, TEXT, TEXT,
    INTEGER, INTEGER, INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_duplicate_privacy_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_admit_v3(
    JSONB, TEXT, TEXT, TEXT, TEXT, TEXT,
    INTEGER, INTEGER, INTEGER, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_admit_v2(
    JSONB, TEXT, TEXT, TEXT, TEXT, TEXT,
    INTEGER, INTEGER, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_duplicate_privacy_contract_v1()
    TO service_role;

COMMENT ON FUNCTION public.research_lab_source_add_admit_v3(
    JSONB, TEXT, TEXT, TEXT, TEXT, TEXT,
    INTEGER, INTEGER, INTEGER, INTEGER
) IS
    'Atomically classifies SOURCE_ADD duplicates before enforcing the durable distinct-source cooldown and v1 admission caps.';
COMMENT ON FUNCTION public.research_lab_source_add_admit_v2(
    JSONB, TEXT, TEXT, TEXT, TEXT, TEXT,
    INTEGER, INTEGER, INTEGER
) IS
    'Rolling-release compatibility wrapper for duplicate-private SOURCE_ADD admission with the production 20-second cooldown.';
COMMENT ON FUNCTION public.research_lab_source_add_duplicate_privacy_contract_v1() IS
    'Read-only exact function-authority, policy, signature, and ACL contract for duplicate-private SOURCE_ADD admission.';

NOTIFY pgrst, 'reload schema';

COMMIT;
