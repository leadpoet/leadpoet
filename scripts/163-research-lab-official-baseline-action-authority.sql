-- Append-only provider/verifier action authority for the official baseline.
--
-- This authority is deliberately independent of measured routing experiments.
-- It stores only release/action identities, protected-job references, hashes,
-- and bounded accounting. Provider responses and company data remain inside
-- the protected execution job and the encrypted terminal/checkpoint stores.
-- A reservation may authorize exactly one new protected job. Expiry never
-- authorizes another POST; an existing reservation can only reconcile the
-- already-bound protected_job_ref or become terminal_uncertain.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_exact_keys_v1(
    p_value JSONB,
    p_keys TEXT[]
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $exact_keys$
    SELECT pg_catalog.jsonb_typeof(p_value) = 'object'
       AND p_value ?& p_keys
       AND (
            SELECT pg_catalog.count(*) = pg_catalog.cardinality(p_keys)
              FROM pg_catalog.jsonb_object_keys(p_value) AS key_name
       )
$exact_keys$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_hash_v1(
    p_value JSONB
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $hash$
    SELECT public.research_lab_routing_jsonb_hash_v2(p_value)
$hash$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_reject_secret_doc_v1(
    p_value JSONB,
    p_label TEXT
)
RETURNS VOID
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $reject_secret$
DECLARE
    key_name TEXT;
BEGIN
    IF pg_catalog.jsonb_typeof(p_value) <> 'object'
       OR pg_catalog.octet_length(p_value::TEXT) > 262144
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_%_invalid', p_label
            USING ERRCODE = '22023';
    END IF;
    FOR key_name IN
        SELECT key_entry.value
          FROM pg_catalog.jsonb_object_keys(p_value) AS key_entry(value)
    LOOP
        IF key_name !~ '_sha256$'
           AND key_name ~* '(^|_)(authorization|cookie|credential|password|secret|token|raw_(request|response|payload|content)|request_body|response_body|provider_response|company_data)($|_)'
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_%_contains_protected_material', p_label
                USING ERRCODE = '22023';
        END IF;
    END LOOP;
END;
$reject_secret$;

CREATE TABLE IF NOT EXISTS public.research_lab_official_baseline_runs_v1 (
    run_sha256                  TEXT PRIMARY KEY CHECK (
        run_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    registration_sha256         TEXT NOT NULL UNIQUE CHECK (
        registration_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    benchmark_date              DATE NOT NULL,
    rolling_window_hash         TEXT NOT NULL CHECK (
        rolling_window_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    model_artifact_hash         TEXT NOT NULL CHECK (
        model_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    manifest_hash               TEXT NOT NULL CHECK (
        manifest_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    release_selection_sha256    TEXT NOT NULL CHECK (
        release_selection_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    artifact_key_sha256         TEXT NOT NULL CHECK (
        artifact_key_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    protocol_generation_sha256  TEXT NOT NULL CHECK (
        protocol_generation_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    projection_identity_sha256  TEXT NOT NULL CHECK (
        projection_identity_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    authority_identity_sha256   TEXT NOT NULL CHECK (
        authority_identity_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    registration_doc            JSONB NOT NULL CHECK (
        pg_catalog.jsonb_typeof(registration_doc) = 'object'
    ),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (
        registration_doc = pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.official_baseline_run_registration.v1',
            'run_sha256', run_sha256,
            'benchmark_date', benchmark_date::TEXT,
            'rolling_window_hash', rolling_window_hash,
            'model_artifact_hash', model_artifact_hash,
            'manifest_hash', manifest_hash,
            'release_selection_sha256', release_selection_sha256,
            'artifact_key_sha256', artifact_key_sha256,
            'protocol_generation_sha256', protocol_generation_sha256,
            'projection_identity_sha256', projection_identity_sha256,
            'authority_identity_sha256', authority_identity_sha256
        )
    ),
    CHECK (
        registration_sha256 =
            public.research_lab_official_baseline_hash_v1(registration_doc)
    )
);

CREATE TABLE IF NOT EXISTS public.research_lab_official_baseline_action_attempts_v1 (
    attempt_key                 TEXT PRIMARY KEY CHECK (
        attempt_key ~ '^sha256:[0-9a-f]{64}$'
    ),
    run_sha256                  TEXT NOT NULL REFERENCES
        public.research_lab_official_baseline_runs_v1(run_sha256),
    unit_ref                    TEXT NOT NULL CHECK (
        unit_ref ~ '^baseline_icp:[0-9a-f]{64}$'
    ),
    action_idempotency_sha256   TEXT NOT NULL CHECK (
        action_idempotency_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    action_sha256               TEXT NOT NULL CHECK (
        action_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    action_sequence             INTEGER NOT NULL CHECK (
        action_sequence BETWEEN 0 AND 9999
    ),
    action_type                 TEXT NOT NULL CHECK (
        action_type IN (
            'normalize_icp',
            'execute_candidate_tool', 'verify_company',
            'execute_intent_tool', 'verify_intent',
            'execute_contact_tool', 'verify_contact'
        )
    ),
    tool_id                     TEXT NOT NULL CHECK (
        tool_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    ),
    binding_contract_sha256     TEXT NOT NULL CHECK (
        binding_contract_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    request_fingerprint_sha256  TEXT NOT NULL CHECK (
        request_fingerprint_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    request_body_sha256         TEXT NOT NULL CHECK (
        request_body_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    call_cap                    INTEGER NOT NULL CHECK (
        call_cap BETWEEN 0 AND 100000
    ),
    credit_cap_microunits       BIGINT NOT NULL CHECK (
        credit_cap_microunits BETWEEN 0 AND 100000000
    ),
    timeout_ms                  INTEGER NOT NULL CHECK (
        timeout_ms BETWEEN 1 AND 900000
    ),
    protected_job_ref           TEXT NOT NULL UNIQUE CHECK (
        protected_job_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    ),
    protected_request_sha256    TEXT NOT NULL CHECK (
        protected_request_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    lease_holder_sha256         TEXT NOT NULL CHECK (
        lease_holder_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    expected_frontier_sha256    TEXT NOT NULL CHECK (
        expected_frontier_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    reservation_ref             TEXT NOT NULL UNIQUE CHECK (
        reservation_ref ~ '^baseline_reservation:[0-9a-f]{64}$'
    ),
    lease_generation            BIGINT NOT NULL DEFAULT 1 CHECK (
        lease_generation = 1
    ),
    lease_expires_at            TIMESTAMPTZ NOT NULL,
    authorization_sha256        TEXT NOT NULL UNIQUE CHECK (
        authorization_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    authorization_doc           JSONB NOT NULL CHECK (
        pg_catalog.jsonb_typeof(authorization_doc) = 'object'
    ),
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    UNIQUE (run_sha256, unit_ref, action_idempotency_sha256),
    UNIQUE (run_sha256, unit_ref, action_sequence),
    CHECK (
        (action_type IN (
            'normalize_icp', 'execute_candidate_tool',
            'execute_intent_tool', 'execute_contact_tool'
        ) AND call_cap >= 1)
        OR
        (action_type IN (
            'verify_company', 'verify_intent', 'verify_contact'
        ) AND call_cap = 0 AND credit_cap_microunits = 0)
    ),
    CHECK (
        authorization_sha256 =
            public.research_lab_official_baseline_hash_v1(authorization_doc)
    )
);

CREATE INDEX IF NOT EXISTS idx_rl_official_baseline_attempt_unit_v1
    ON public.research_lab_official_baseline_action_attempts_v1(
        run_sha256, unit_ref, action_sequence
    );

CREATE TABLE IF NOT EXISTS public.research_lab_official_baseline_action_terminals_v1 (
    attempt_key                         TEXT PRIMARY KEY REFERENCES
        public.research_lab_official_baseline_action_attempts_v1(attempt_key),
    terminal_state                      TEXT NOT NULL CHECK (
        terminal_state IN ('terminal_known', 'terminal_uncertain')
    ),
    reservation_ref                     TEXT NOT NULL CHECK (
        reservation_ref ~ '^baseline_reservation:[0-9a-f]{64}$'
    ),
    lease_generation                    BIGINT NOT NULL CHECK (
        lease_generation = 1
    ),
    protected_job_ref                   TEXT NOT NULL CHECK (
        protected_job_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    ),
    protected_request_sha256            TEXT NOT NULL CHECK (
        protected_request_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    protected_result_sha256             TEXT CHECK (
        protected_result_sha256 IS NULL
        OR protected_result_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    protected_terminal_receipt_ref      TEXT CHECK (
        protected_terminal_receipt_ref IS NULL
        OR protected_terminal_receipt_ref
            ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    ),
    protected_terminal_receipt_sha256   TEXT CHECK (
        protected_terminal_receipt_sha256 IS NULL
        OR protected_terminal_receipt_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    provider_request_ref                TEXT CHECK (
        provider_request_ref IS NULL
        OR provider_request_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    ),
    provider_receipt_ref                TEXT CHECK (
        provider_receipt_ref IS NULL
        OR provider_receipt_ref ~ '^provider_receipt:[0-9a-f]{16}$'
    ),
    provider_receipt_sha256             TEXT CHECK (
        provider_receipt_sha256 IS NULL
        OR provider_receipt_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    provider_identity_sha256            TEXT CHECK (
        provider_identity_sha256 IS NULL
        OR provider_identity_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    model_provider_response_sha256      TEXT CHECK (
        model_provider_response_sha256 IS NULL
        OR model_provider_response_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    outcome                             TEXT CHECK (
        outcome IS NULL OR outcome IN ('succeeded', 'empty', 'failed')
    ),
    call_count                          INTEGER CHECK (
        call_count IS NULL OR call_count BETWEEN 0 AND 100000
    ),
    cost_microunits                     BIGINT CHECK (
        cost_microunits IS NULL OR cost_microunits BETWEEN 0 AND 100000000
    ),
    latency_ms                          BIGINT CHECK (
        latency_ms IS NULL OR latency_ms BETWEEN 0 AND 900000
    ),
    uncertainty_sha256                 TEXT CHECK (
        uncertainty_sha256 IS NULL
        OR uncertainty_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    terminal_doc_sha256                TEXT NOT NULL UNIQUE CHECK (
        terminal_doc_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    terminal_doc                       JSONB NOT NULL CHECK (
        pg_catalog.jsonb_typeof(terminal_doc) = 'object'
    ),
    terminal_attempt_sha256            TEXT NOT NULL UNIQUE CHECK (
        terminal_attempt_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    created_at                          TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (
        terminal_doc_sha256 =
            public.research_lab_official_baseline_hash_v1(terminal_doc)
    ),
    CHECK (
        (terminal_state = 'terminal_known'
         AND protected_result_sha256 IS NOT NULL
         AND protected_terminal_receipt_ref IS NOT NULL
         AND protected_terminal_receipt_sha256 IS NOT NULL
         AND model_provider_response_sha256 IS NOT NULL
         AND outcome IS NOT NULL
         AND call_count IS NOT NULL
         AND cost_microunits IS NOT NULL
         AND latency_ms IS NOT NULL
         AND uncertainty_sha256 IS NULL)
        OR
        (terminal_state = 'terminal_uncertain'
         AND protected_result_sha256 IS NULL
         AND protected_terminal_receipt_ref IS NULL
         AND protected_terminal_receipt_sha256 IS NULL
         AND provider_receipt_ref IS NULL
         AND provider_receipt_sha256 IS NULL
         AND provider_identity_sha256 IS NULL
         AND model_provider_response_sha256 IS NULL
         AND outcome IS NULL
         AND call_count IS NULL
         AND cost_microunits IS NULL
         AND latency_ms IS NULL
         AND uncertainty_sha256 IS NOT NULL)
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_official_baseline_provider_request_v1
    ON public.research_lab_official_baseline_action_terminals_v1(
        provider_request_ref
    ) WHERE provider_request_ref IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_official_baseline_provider_receipt_v1
    ON public.research_lab_official_baseline_action_terminals_v1(
        provider_receipt_ref
    ) WHERE provider_receipt_ref IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.research_lab_official_baseline_unit_closures_v1 (
    closure_ref                TEXT PRIMARY KEY CHECK (
        closure_ref ~ '^baseline_closure:[0-9a-f]{64}$'
    ),
    closure_sha256             TEXT NOT NULL UNIQUE CHECK (
        closure_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    run_sha256                 TEXT NOT NULL REFERENCES
        public.research_lab_official_baseline_runs_v1(run_sha256),
    unit_ref                   TEXT NOT NULL CHECK (
        unit_ref ~ '^baseline_icp:[0-9a-f]{64}$'
    ),
    protocol_generation_sha256 TEXT NOT NULL CHECK (
        protocol_generation_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    raw_input_sha256          TEXT NOT NULL CHECK (
        raw_input_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    start_request_sha256      TEXT NOT NULL CHECK (
        start_request_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    terminal_result_sha256    TEXT NOT NULL CHECK (
        terminal_result_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    model_receipt_sha256      TEXT NOT NULL CHECK (
        model_receipt_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    projection_sha256         TEXT NOT NULL CHECK (
        projection_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    ordered_attempt_keys      JSONB NOT NULL CHECK (
        pg_catalog.jsonb_typeof(ordered_attempt_keys) = 'array'
        AND pg_catalog.jsonb_array_length(ordered_attempt_keys) BETWEEN 1 AND 10000
    ),
    ordered_attempt_sha256s   JSONB NOT NULL CHECK (
        pg_catalog.jsonb_typeof(ordered_attempt_sha256s) = 'array'
        AND pg_catalog.jsonb_array_length(ordered_attempt_sha256s)
            = pg_catalog.jsonb_array_length(ordered_attempt_keys)
    ),
    provider_frontier_sha256  TEXT NOT NULL CHECK (
        provider_frontier_sha256 ~ '^sha256:[0-9a-f]{64}$'
    ),
    closure_doc               JSONB NOT NULL CHECK (
        pg_catalog.jsonb_typeof(closure_doc) = 'object'
    ),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    UNIQUE (run_sha256, unit_ref),
    CHECK (
        closure_sha256 =
            public.research_lab_official_baseline_hash_v1(closure_doc)
    )
);

CREATE OR REPLACE FUNCTION public.prevent_research_lab_official_baseline_mutation_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $no_mutation$
BEGIN
    RAISE EXCEPTION '% is append-only; write a new immutable row', TG_TABLE_NAME;
END;
$no_mutation$;

DO $append_only_triggers$
DECLARE
    relation_name TEXT;
    trigger_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_official_baseline_runs_v1',
        'research_lab_official_baseline_action_attempts_v1',
        'research_lab_official_baseline_action_terminals_v1',
        'research_lab_official_baseline_unit_closures_v1'
    ] LOOP
        trigger_name := CASE relation_name
            WHEN 'research_lab_official_baseline_runs_v1'
                THEN 'trg_rl_ob_runs_no_mutation'
            WHEN 'research_lab_official_baseline_action_attempts_v1'
                THEN 'trg_rl_ob_attempts_no_mutation'
            WHEN 'research_lab_official_baseline_action_terminals_v1'
                THEN 'trg_rl_ob_terminals_no_mutation'
            ELSE 'trg_rl_ob_closures_no_mutation'
        END;
        IF NOT EXISTS (
            SELECT 1
              FROM pg_catalog.pg_trigger trigger_meta
              JOIN pg_catalog.pg_class relation_meta
                ON relation_meta.oid = trigger_meta.tgrelid
              JOIN pg_catalog.pg_namespace namespace_meta
                ON namespace_meta.oid = relation_meta.relnamespace
             WHERE namespace_meta.nspname = 'public'
               AND relation_meta.relname = relation_name
               AND trigger_meta.tgname = trigger_name
               AND NOT trigger_meta.tgisinternal
        ) THEN
            EXECUTE pg_catalog.format(
                'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I '
                'FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_official_baseline_mutation_v1()',
                trigger_name,
                relation_name
            );
        END IF;
    END LOOP;
END;
$append_only_triggers$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_provider_frontier_doc_v1(
    p_run_sha256 TEXT,
    p_unit_ref TEXT
)
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $provider_frontier_doc$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_provider_frontier.v1',
        'ordered_attempt_keys',
            COALESCE(
                pg_catalog.jsonb_agg(
                    attempt.attempt_key ORDER BY attempt.action_sequence
                ) FILTER (WHERE attempt.attempt_key IS NOT NULL),
                '[]'::JSONB
            ),
        'ordered_attempt_sha256s',
            COALESCE(
                pg_catalog.jsonb_agg(
                    terminal.terminal_attempt_sha256
                    ORDER BY attempt.action_sequence
                ) FILTER (WHERE attempt.attempt_key IS NOT NULL),
                '[]'::JSONB
            )
    )
      FROM public.research_lab_official_baseline_action_attempts_v1 attempt
      JOIN public.research_lab_official_baseline_action_terminals_v1 terminal
        ON terminal.attempt_key = attempt.attempt_key
     WHERE attempt.run_sha256 = p_run_sha256
       AND attempt.unit_ref = p_unit_ref
$provider_frontier_doc$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_register_run_v1(
    p_registration JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $register_run$
DECLARE
    registration_hash TEXT;
    existing public.research_lab_official_baseline_runs_v1%ROWTYPE;
    inserted BOOLEAN := FALSE;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_registration,
        ARRAY[
            'schema_version', 'run_sha256', 'benchmark_date',
            'rolling_window_hash', 'model_artifact_hash', 'manifest_hash',
            'release_selection_sha256', 'artifact_key_sha256',
            'protocol_generation_sha256', 'projection_identity_sha256',
            'authority_identity_sha256'
        ]
    )
       OR p_registration->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_run_registration.v1'
       OR p_registration->>'run_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'benchmark_date' !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
       OR p_registration->>'rolling_window_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'model_artifact_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'manifest_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'release_selection_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'artifact_key_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'protocol_generation_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'projection_identity_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_registration->>'authority_identity_sha256' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_run_registration_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_registration, 'run_registration'
    );
    registration_hash :=
        public.research_lab_official_baseline_hash_v1(p_registration);

    INSERT INTO public.research_lab_official_baseline_runs_v1 (
        run_sha256, registration_sha256, benchmark_date,
        rolling_window_hash, model_artifact_hash, manifest_hash,
        release_selection_sha256, artifact_key_sha256,
        protocol_generation_sha256, projection_identity_sha256,
        authority_identity_sha256, registration_doc
    ) VALUES (
        p_registration->>'run_sha256', registration_hash,
        (p_registration->>'benchmark_date')::DATE,
        p_registration->>'rolling_window_hash',
        p_registration->>'model_artifact_hash',
        p_registration->>'manifest_hash',
        p_registration->>'release_selection_sha256',
        p_registration->>'artifact_key_sha256',
        p_registration->>'protocol_generation_sha256',
        p_registration->>'projection_identity_sha256',
        p_registration->>'authority_identity_sha256',
        p_registration
    ) ON CONFLICT DO NOTHING
    RETURNING * INTO existing;
    inserted := FOUND;

    IF NOT inserted THEN
        SELECT * INTO existing
          FROM public.research_lab_official_baseline_runs_v1 run
         WHERE run.run_sha256 = p_registration->>'run_sha256';
        IF NOT FOUND OR existing.registration_doc IS DISTINCT FROM p_registration
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_run_registration_conflict'
                USING ERRCODE = '23505';
        END IF;
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_run_registration_result.v1',
        'run_sha256', existing.run_sha256,
        'registration_sha256', existing.registration_sha256,
        'idempotent', NOT inserted
    );
END;
$register_run$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_reserve_action_v1(
    p_authorization JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $reserve_action$
DECLARE
    authorization_hash TEXT;
    frontier_doc JSONB;
    frontier_hash TEXT;
    expected_sequence INTEGER;
    reservation_ref_value TEXT;
    expires_at_value TIMESTAMPTZ;
    attempt public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
    terminal public.research_lab_official_baseline_action_terminals_v1%ROWTYPE;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_authorization,
        ARRAY[
            'schema_version', 'attempt_key', 'run_sha256', 'unit_ref',
            'action_idempotency_sha256', 'action_sha256', 'action_sequence',
            'action_type', 'tool_id', 'binding_contract_sha256',
            'request_fingerprint_sha256', 'request_body_sha256', 'call_cap',
            'credit_cap_microunits', 'timeout_ms', 'protected_job_ref',
            'protected_request_sha256', 'lease_holder_sha256',
            'expected_frontier_sha256'
        ]
    )
       OR p_authorization->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_action_authorization.v1'
       OR p_authorization->>'attempt_key' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'run_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'unit_ref' !~ '^baseline_icp:[0-9a-f]{64}$'
       OR p_authorization->>'action_idempotency_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'action_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_authorization->'action_sequence') IS DISTINCT FROM 'number'
       OR (p_authorization->>'action_sequence') !~ '^[0-9]{1,4}$'
       OR (p_authorization->>'action_sequence')::INTEGER NOT BETWEEN 0 AND 9999
       OR p_authorization->>'action_type' NOT IN (
            'normalize_icp', 'execute_candidate_tool', 'verify_company',
            'execute_intent_tool', 'verify_intent',
            'execute_contact_tool', 'verify_contact'
       )
       OR p_authorization->>'tool_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_authorization->>'binding_contract_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'request_fingerprint_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'request_body_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_authorization->'call_cap') IS DISTINCT FROM 'number'
       OR (p_authorization->>'call_cap') !~ '^[0-9]{1,6}$'
       OR (p_authorization->>'call_cap')::INTEGER NOT BETWEEN 0 AND 100000
       OR pg_catalog.jsonb_typeof(p_authorization->'credit_cap_microunits') IS DISTINCT FROM 'number'
       OR (p_authorization->>'credit_cap_microunits') !~ '^[0-9]{1,9}$'
       OR (p_authorization->>'credit_cap_microunits')::BIGINT NOT BETWEEN 0 AND 100000000
       OR pg_catalog.jsonb_typeof(p_authorization->'timeout_ms') IS DISTINCT FROM 'number'
       OR (p_authorization->>'timeout_ms') !~ '^[1-9][0-9]{0,5}$'
       OR (p_authorization->>'timeout_ms')::INTEGER NOT BETWEEN 1 AND 900000
       OR p_authorization->>'protected_job_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_authorization->>'protected_request_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'lease_holder_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization->>'expected_frontier_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR (
            p_authorization->>'action_type' IN (
                'verify_company', 'verify_intent', 'verify_contact'
            )
            AND (
                (p_authorization->>'call_cap')::INTEGER <> 0
                OR (p_authorization->>'credit_cap_microunits')::BIGINT <> 0
            )
       )
       OR (
            p_authorization->>'action_type' IN (
                'normalize_icp', 'execute_candidate_tool',
                'execute_intent_tool', 'execute_contact_tool'
            )
            AND (p_authorization->>'call_cap')::INTEGER < 1
       )
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_authorization_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_authorization, 'action_authorization'
    );
    authorization_hash :=
        public.research_lab_official_baseline_hash_v1(p_authorization);

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            (p_authorization->>'run_sha256') || ':' ||
                (p_authorization->>'unit_ref'),
            0
        )
    );
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_runs_v1 run
         WHERE run.run_sha256 = p_authorization->>'run_sha256'
    ) THEN
        RAISE EXCEPTION 'research_lab_official_baseline_run_not_registered'
            USING ERRCODE = '23503';
    END IF;

    SELECT * INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 current_attempt
     WHERE current_attempt.attempt_key = p_authorization->>'attempt_key';
    IF FOUND THEN
        IF attempt.authorization_doc IS DISTINCT FROM p_authorization THEN
            RAISE EXCEPTION 'research_lab_official_baseline_action_authorization_conflict'
                USING ERRCODE = '23505';
        END IF;
        SELECT * INTO terminal
          FROM public.research_lab_official_baseline_action_terminals_v1 current_terminal
         WHERE current_terminal.attempt_key = attempt.attempt_key;
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_reservation_result.v1',
            'disposition', CASE
                WHEN FOUND THEN terminal.terminal_state
                ELSE 'reserved_existing'
            END,
            'attempt_key', attempt.attempt_key,
            'reservation_ref', attempt.reservation_ref,
            'lease_generation', attempt.lease_generation,
            'lease_expires_at', attempt.lease_expires_at,
            'protected_job_ref', attempt.protected_job_ref,
            'protected_request_sha256', attempt.protected_request_sha256,
            'attempt_sha256', CASE
                WHEN FOUND THEN terminal.terminal_attempt_sha256
                ELSE attempt.authorization_sha256
            END
        );
    END IF;

    IF EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_unit_closures_v1 closure
         WHERE closure.run_sha256 = p_authorization->>'run_sha256'
           AND closure.unit_ref = p_authorization->>'unit_ref'
    ) THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_already_closed'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_official_baseline_action_attempts_v1 prior_attempt
          JOIN public.research_lab_official_baseline_action_terminals_v1 prior_terminal
            ON prior_terminal.attempt_key = prior_attempt.attempt_key
         WHERE prior_attempt.run_sha256 = p_authorization->>'run_sha256'
           AND prior_attempt.unit_ref = p_authorization->>'unit_ref'
           AND prior_terminal.terminal_state = 'terminal_uncertain'
    ) THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_terminal_uncertain'
            USING ERRCODE = '40003';
    END IF;

    SELECT prior_attempt.* INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 prior_attempt
      LEFT JOIN public.research_lab_official_baseline_action_terminals_v1 prior_terminal
        ON prior_terminal.attempt_key = prior_attempt.attempt_key
     WHERE prior_attempt.run_sha256 = p_authorization->>'run_sha256'
       AND prior_attempt.unit_ref = p_authorization->>'unit_ref'
       AND prior_terminal.attempt_key IS NULL
     ORDER BY prior_attempt.action_sequence
     LIMIT 1;
    IF FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_reservation_result.v1',
            'disposition', 'inflight',
            'attempt_key', attempt.attempt_key,
            'reservation_ref', attempt.reservation_ref,
            'lease_generation', attempt.lease_generation,
            'lease_expires_at', attempt.lease_expires_at,
            'protected_job_ref', attempt.protected_job_ref,
            'protected_request_sha256', attempt.protected_request_sha256,
            'attempt_sha256', attempt.authorization_sha256
        );
    END IF;

    frontier_doc := public.research_lab_official_baseline_provider_frontier_doc_v1(
        p_authorization->>'run_sha256', p_authorization->>'unit_ref'
    );
    frontier_hash := public.research_lab_official_baseline_hash_v1(frontier_doc);
    IF p_authorization->>'expected_frontier_sha256' IS DISTINCT FROM frontier_hash
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_provider_frontier_conflict'
            USING ERRCODE = '40001';
    END IF;
    SELECT COALESCE(MAX(prior_attempt.action_sequence) + 1, 0)
      INTO expected_sequence
      FROM public.research_lab_official_baseline_action_attempts_v1 prior_attempt
     WHERE prior_attempt.run_sha256 = p_authorization->>'run_sha256'
       AND prior_attempt.unit_ref = p_authorization->>'unit_ref';
    IF (p_authorization->>'action_sequence')::INTEGER <> expected_sequence THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_sequence_conflict'
            USING ERRCODE = '40001';
    END IF;

    reservation_ref_value := 'baseline_reservation:' ||
        pg_catalog.substr(p_authorization->>'attempt_key', 8);
    expires_at_value := pg_catalog.clock_timestamp() + pg_catalog.make_interval(
        secs => ((p_authorization->>'timeout_ms')::INTEGER + 999) / 1000 + 60
    );
    INSERT INTO public.research_lab_official_baseline_action_attempts_v1 (
        attempt_key, run_sha256, unit_ref, action_idempotency_sha256,
        action_sha256, action_sequence, action_type, tool_id,
        binding_contract_sha256, request_fingerprint_sha256,
        request_body_sha256, call_cap, credit_cap_microunits, timeout_ms,
        protected_job_ref, protected_request_sha256, lease_holder_sha256,
        expected_frontier_sha256, reservation_ref, lease_expires_at,
        authorization_sha256, authorization_doc
    ) VALUES (
        p_authorization->>'attempt_key', p_authorization->>'run_sha256',
        p_authorization->>'unit_ref',
        p_authorization->>'action_idempotency_sha256',
        p_authorization->>'action_sha256',
        (p_authorization->>'action_sequence')::INTEGER,
        p_authorization->>'action_type', p_authorization->>'tool_id',
        p_authorization->>'binding_contract_sha256',
        p_authorization->>'request_fingerprint_sha256',
        p_authorization->>'request_body_sha256',
        (p_authorization->>'call_cap')::INTEGER,
        (p_authorization->>'credit_cap_microunits')::BIGINT,
        (p_authorization->>'timeout_ms')::INTEGER,
        p_authorization->>'protected_job_ref',
        p_authorization->>'protected_request_sha256',
        p_authorization->>'lease_holder_sha256',
        p_authorization->>'expected_frontier_sha256',
        reservation_ref_value, expires_at_value,
        authorization_hash, p_authorization
    ) RETURNING * INTO attempt;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_reservation_result.v1',
        'disposition', 'reserved_new',
        'attempt_key', attempt.attempt_key,
        'reservation_ref', attempt.reservation_ref,
        'lease_generation', attempt.lease_generation,
        'lease_expires_at', attempt.lease_expires_at,
        'protected_job_ref', attempt.protected_job_ref,
        'protected_request_sha256', attempt.protected_request_sha256,
        'attempt_sha256', attempt.authorization_sha256
    );
END;
$reserve_action$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_record_terminal_known_v1(
    p_terminal JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $record_terminal_known$
DECLARE
    attempt public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
    existing public.research_lab_official_baseline_action_terminals_v1%ROWTYPE;
    terminal_doc_hash TEXT;
    terminal_attempt_hash TEXT;
    is_verifier BOOLEAN;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_terminal,
        ARRAY[
            'schema_version', 'attempt_key', 'reservation_ref',
            'lease_generation', 'protected_job_ref',
            'protected_request_sha256', 'protected_result_sha256',
            'protected_terminal_receipt_ref',
            'protected_terminal_receipt_sha256', 'provider_request_ref',
            'provider_receipt_ref', 'provider_receipt_sha256',
            'provider_identity_sha256', 'model_provider_response_sha256',
            'outcome', 'call_count', 'cost_microunits', 'latency_ms'
        ]
    )
       OR p_terminal->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_action_terminal_known.v1'
       OR p_terminal->>'attempt_key' !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal->>'reservation_ref' !~ '^baseline_reservation:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_terminal->'lease_generation') IS DISTINCT FROM 'number'
       OR p_terminal->>'lease_generation' IS DISTINCT FROM '1'
       OR p_terminal->>'protected_job_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_terminal->>'protected_request_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal->>'protected_result_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal->>'protected_terminal_receipt_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_terminal->>'protected_terminal_receipt_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal->>'model_provider_response_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal->>'outcome' IS NULL
       OR p_terminal->>'outcome' NOT IN ('succeeded', 'empty', 'failed')
       OR pg_catalog.jsonb_typeof(p_terminal->'call_count') IS DISTINCT FROM 'number'
       OR (p_terminal->>'call_count') !~ '^[0-9]{1,6}$'
       OR (p_terminal->>'call_count')::INTEGER NOT BETWEEN 0 AND 100000
       OR pg_catalog.jsonb_typeof(p_terminal->'cost_microunits') IS DISTINCT FROM 'number'
       OR (p_terminal->>'cost_microunits') !~ '^[0-9]{1,9}$'
       OR (p_terminal->>'cost_microunits')::BIGINT NOT BETWEEN 0 AND 100000000
       OR pg_catalog.jsonb_typeof(p_terminal->'latency_ms') IS DISTINCT FROM 'number'
       OR (p_terminal->>'latency_ms') !~ '^[0-9]{1,6}$'
       OR (p_terminal->>'latency_ms')::BIGINT NOT BETWEEN 0 AND 900000
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_terminal_known_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_terminal, 'terminal_known'
    );

    SELECT * INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 current_attempt
     WHERE current_attempt.attempt_key = p_terminal->>'attempt_key'
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_reservation_missing'
            USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_official_baseline_action_terminals_v1 current_terminal
     WHERE current_terminal.attempt_key = attempt.attempt_key;
    IF FOUND THEN
        IF existing.terminal_state IS DISTINCT FROM 'terminal_known'
           OR existing.terminal_doc IS DISTINCT FROM p_terminal
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_terminal_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_terminal_result.v1',
            'state', existing.terminal_state,
            'attempt_key', existing.attempt_key,
            'attempt_sha256', existing.terminal_attempt_sha256,
            'idempotent', TRUE
        );
    END IF;
    IF p_terminal->>'reservation_ref' IS DISTINCT FROM attempt.reservation_ref
       OR (p_terminal->>'lease_generation')::BIGINT IS DISTINCT FROM attempt.lease_generation
       OR p_terminal->>'protected_job_ref' IS DISTINCT FROM attempt.protected_job_ref
       OR p_terminal->>'protected_request_sha256' IS DISTINCT FROM attempt.protected_request_sha256
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_terminal_reservation_fence_conflict'
            USING ERRCODE = '40001';
    END IF;

    is_verifier := attempt.action_type IN (
        'verify_company', 'verify_intent', 'verify_contact'
    );
    IF is_verifier THEN
        IF p_terminal->>'outcome' IS NULL
           OR p_terminal->>'outcome' NOT IN ('succeeded', 'failed')
           OR (p_terminal->>'call_count')::INTEGER <> 0
           OR (p_terminal->>'cost_microunits')::BIGINT <> 0
           OR p_terminal->'provider_request_ref' IS DISTINCT FROM 'null'::JSONB
           OR p_terminal->'provider_receipt_ref' IS DISTINCT FROM 'null'::JSONB
           OR p_terminal->'provider_receipt_sha256' IS DISTINCT FROM 'null'::JSONB
           OR p_terminal->'provider_identity_sha256' IS DISTINCT FROM 'null'::JSONB
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_verifier_accounting_invalid'
                USING ERRCODE = '22023';
        END IF;
    ELSE
        IF pg_catalog.jsonb_typeof(p_terminal->'provider_request_ref')
                IS DISTINCT FROM 'string'
           OR pg_catalog.jsonb_typeof(p_terminal->'provider_receipt_ref')
                IS DISTINCT FROM 'string'
           OR pg_catalog.jsonb_typeof(p_terminal->'provider_receipt_sha256')
                IS DISTINCT FROM 'string'
           OR pg_catalog.jsonb_typeof(p_terminal->'provider_identity_sha256')
                IS DISTINCT FROM 'string'
           OR p_terminal->>'provider_request_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_terminal->>'provider_receipt_ref' !~ '^provider_receipt:[0-9a-f]{16}$'
           OR p_terminal->>'provider_receipt_sha256' !~ '^sha256:[0-9a-f]{64}$'
           OR p_terminal->>'provider_identity_sha256' !~ '^sha256:[0-9a-f]{64}$'
           OR (p_terminal->>'call_count')::INTEGER < 1
           OR (p_terminal->>'call_count')::INTEGER > attempt.call_cap
           OR (p_terminal->>'cost_microunits')::BIGINT > attempt.credit_cap_microunits
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_provider_accounting_invalid'
                USING ERRCODE = '22023';
        END IF;
    END IF;

    terminal_doc_hash :=
        public.research_lab_official_baseline_hash_v1(p_terminal);
    terminal_attempt_hash := public.research_lab_official_baseline_hash_v1(
        pg_catalog.jsonb_build_object(
            'authorization_sha256', attempt.authorization_sha256,
            'terminal_state', 'terminal_known',
            'terminal_doc_sha256', terminal_doc_hash
        )
    );
    INSERT INTO public.research_lab_official_baseline_action_terminals_v1 (
        attempt_key, terminal_state, reservation_ref, lease_generation,
        protected_job_ref, protected_request_sha256, protected_result_sha256,
        protected_terminal_receipt_ref, protected_terminal_receipt_sha256,
        provider_request_ref, provider_receipt_ref, provider_receipt_sha256,
        provider_identity_sha256, model_provider_response_sha256, outcome,
        call_count, cost_microunits, latency_ms, terminal_doc_sha256,
        terminal_doc, terminal_attempt_sha256
    ) VALUES (
        attempt.attempt_key, 'terminal_known', attempt.reservation_ref,
        attempt.lease_generation, attempt.protected_job_ref,
        attempt.protected_request_sha256,
        p_terminal->>'protected_result_sha256',
        p_terminal->>'protected_terminal_receipt_ref',
        p_terminal->>'protected_terminal_receipt_sha256',
        p_terminal->>'provider_request_ref',
        p_terminal->>'provider_receipt_ref',
        p_terminal->>'provider_receipt_sha256',
        p_terminal->>'provider_identity_sha256',
        p_terminal->>'model_provider_response_sha256',
        p_terminal->>'outcome', (p_terminal->>'call_count')::INTEGER,
        (p_terminal->>'cost_microunits')::BIGINT,
        (p_terminal->>'latency_ms')::BIGINT,
        terminal_doc_hash, p_terminal, terminal_attempt_hash
    ) RETURNING * INTO existing;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_terminal_result.v1',
        'state', existing.terminal_state,
        'attempt_key', existing.attempt_key,
        'attempt_sha256', existing.terminal_attempt_sha256,
        'idempotent', FALSE
    );
END;
$record_terminal_known$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_record_terminal_uncertain_v1(
    p_terminal JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $record_terminal_uncertain$
DECLARE
    attempt public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
    existing public.research_lab_official_baseline_action_terminals_v1%ROWTYPE;
    terminal_doc_hash TEXT;
    terminal_attempt_hash TEXT;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_terminal,
        ARRAY[
            'schema_version', 'attempt_key', 'reservation_ref',
            'lease_generation', 'protected_job_ref',
            'protected_request_sha256', 'provider_request_ref',
            'uncertainty_sha256'
        ]
    )
       OR p_terminal->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_action_terminal_uncertain.v1'
       OR p_terminal->>'attempt_key' !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal->>'reservation_ref' !~ '^baseline_reservation:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_terminal->'lease_generation') IS DISTINCT FROM 'number'
       OR p_terminal->>'lease_generation' IS DISTINCT FROM '1'
       OR p_terminal->>'protected_job_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_terminal->>'protected_request_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR NOT (
            p_terminal->'provider_request_ref' = 'null'::JSONB
            OR p_terminal->>'provider_request_ref'
                ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       )
       OR p_terminal->>'uncertainty_sha256' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_terminal_uncertain_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_terminal, 'terminal_uncertain'
    );

    SELECT * INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 current_attempt
     WHERE current_attempt.attempt_key = p_terminal->>'attempt_key'
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_reservation_missing'
            USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_official_baseline_action_terminals_v1 current_terminal
     WHERE current_terminal.attempt_key = attempt.attempt_key;
    IF FOUND THEN
        IF existing.terminal_state IS DISTINCT FROM 'terminal_uncertain'
           OR existing.terminal_doc IS DISTINCT FROM p_terminal
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_terminal_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_terminal_result.v1',
            'state', existing.terminal_state,
            'attempt_key', existing.attempt_key,
            'attempt_sha256', existing.terminal_attempt_sha256,
            'idempotent', TRUE
        );
    END IF;
    IF p_terminal->>'reservation_ref' IS DISTINCT FROM attempt.reservation_ref
       OR (p_terminal->>'lease_generation')::BIGINT IS DISTINCT FROM attempt.lease_generation
       OR p_terminal->>'protected_job_ref' IS DISTINCT FROM attempt.protected_job_ref
       OR p_terminal->>'protected_request_sha256' IS DISTINCT FROM attempt.protected_request_sha256
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_terminal_reservation_fence_conflict'
            USING ERRCODE = '40001';
    END IF;

    terminal_doc_hash :=
        public.research_lab_official_baseline_hash_v1(p_terminal);
    terminal_attempt_hash := public.research_lab_official_baseline_hash_v1(
        pg_catalog.jsonb_build_object(
            'authorization_sha256', attempt.authorization_sha256,
            'terminal_state', 'terminal_uncertain',
            'terminal_doc_sha256', terminal_doc_hash
        )
    );
    INSERT INTO public.research_lab_official_baseline_action_terminals_v1 (
        attempt_key, terminal_state, reservation_ref, lease_generation,
        protected_job_ref, protected_request_sha256, provider_request_ref,
        uncertainty_sha256, terminal_doc_sha256, terminal_doc,
        terminal_attempt_sha256
    ) VALUES (
        attempt.attempt_key, 'terminal_uncertain', attempt.reservation_ref,
        attempt.lease_generation, attempt.protected_job_ref,
        attempt.protected_request_sha256,
        NULLIF(p_terminal->>'provider_request_ref', ''),
        p_terminal->>'uncertainty_sha256', terminal_doc_hash, p_terminal,
        terminal_attempt_hash
    ) RETURNING * INTO existing;

    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_terminal_result.v1',
        'state', existing.terminal_state,
        'attempt_key', existing.attempt_key,
        'attempt_sha256', existing.terminal_attempt_sha256,
        'idempotent', FALSE
    );
END;
$record_terminal_uncertain$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_load_replay_v1(
    p_identity JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $load_replay$
DECLARE
    attempt public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
    terminal public.research_lab_official_baseline_action_terminals_v1%ROWTYPE;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_identity,
        ARRAY[
            'schema_version', 'attempt_key', 'run_sha256', 'unit_ref',
            'action_idempotency_sha256', 'action_sha256',
            'request_fingerprint_sha256'
        ]
    )
       OR p_identity->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_action_replay_identity.v1'
       OR p_identity->>'attempt_key' !~ '^sha256:[0-9a-f]{64}$'
       OR p_identity->>'run_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_identity->>'unit_ref' !~ '^baseline_icp:[0-9a-f]{64}$'
       OR p_identity->>'action_idempotency_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_identity->>'action_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_identity->>'request_fingerprint_sha256' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_replay_identity_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_identity, 'action_replay_identity'
    );

    SELECT * INTO attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 current_attempt
     WHERE current_attempt.attempt_key = p_identity->>'attempt_key';
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'schema_version',
                'leadpoet.research_lab.official_baseline_action_replay_result.v1',
            'state', 'absent',
            'attempt_key', p_identity->>'attempt_key',
            'reservation_ref', NULL,
            'lease_generation', NULL,
            'lease_expires_at', NULL,
            'protected_job_ref', NULL,
            'protected_request_sha256', NULL,
            'protected_result_sha256', NULL,
            'protected_terminal_receipt_ref', NULL,
            'protected_terminal_receipt_sha256', NULL,
            'provider_request_ref', NULL,
            'provider_receipt_ref', NULL,
            'provider_receipt_sha256', NULL,
            'provider_identity_sha256', NULL,
            'model_provider_response_sha256', NULL,
            'outcome', NULL,
            'call_count', NULL,
            'cost_microunits', NULL,
            'latency_ms', NULL,
            'attempt_sha256', NULL
        );
    END IF;
    IF attempt.run_sha256 IS DISTINCT FROM p_identity->>'run_sha256'
       OR attempt.unit_ref IS DISTINCT FROM p_identity->>'unit_ref'
       OR attempt.action_idempotency_sha256 IS DISTINCT FROM
            p_identity->>'action_idempotency_sha256'
       OR attempt.action_sha256 IS DISTINCT FROM p_identity->>'action_sha256'
       OR attempt.request_fingerprint_sha256 IS DISTINCT FROM
            p_identity->>'request_fingerprint_sha256'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_action_replay_identity_conflict'
            USING ERRCODE = '23505';
    END IF;

    SELECT * INTO terminal
      FROM public.research_lab_official_baseline_action_terminals_v1 current_terminal
     WHERE current_terminal.attempt_key = attempt.attempt_key;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_action_replay_result.v1',
        'state', CASE WHEN FOUND THEN terminal.terminal_state ELSE 'reserved' END,
        'attempt_key', attempt.attempt_key,
        'reservation_ref', attempt.reservation_ref,
        'lease_generation', attempt.lease_generation,
        'lease_expires_at', attempt.lease_expires_at,
        'protected_job_ref', attempt.protected_job_ref,
        'protected_request_sha256', attempt.protected_request_sha256,
        'protected_result_sha256', CASE WHEN FOUND THEN terminal.protected_result_sha256 ELSE NULL END,
        'protected_terminal_receipt_ref', CASE WHEN FOUND THEN terminal.protected_terminal_receipt_ref ELSE NULL END,
        'protected_terminal_receipt_sha256', CASE WHEN FOUND THEN terminal.protected_terminal_receipt_sha256 ELSE NULL END,
        'provider_request_ref', CASE WHEN FOUND THEN terminal.provider_request_ref ELSE NULL END,
        'provider_receipt_ref', CASE WHEN FOUND THEN terminal.provider_receipt_ref ELSE NULL END,
        'provider_receipt_sha256', CASE WHEN FOUND THEN terminal.provider_receipt_sha256 ELSE NULL END,
        'provider_identity_sha256', CASE WHEN FOUND THEN terminal.provider_identity_sha256 ELSE NULL END,
        'model_provider_response_sha256', CASE WHEN FOUND THEN terminal.model_provider_response_sha256 ELSE NULL END,
        'outcome', CASE WHEN FOUND THEN terminal.outcome ELSE NULL END,
        'call_count', CASE WHEN FOUND THEN terminal.call_count ELSE NULL END,
        'cost_microunits', CASE WHEN FOUND THEN terminal.cost_microunits ELSE NULL END,
        'latency_ms', CASE WHEN FOUND THEN terminal.latency_ms ELSE NULL END,
        'attempt_sha256', CASE
            WHEN FOUND THEN terminal.terminal_attempt_sha256
            ELSE attempt.authorization_sha256
        END
    );
END;
$load_replay$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_close_unit_v1(
    p_completion JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $close_unit$
DECLARE
    registered public.research_lab_official_baseline_runs_v1%ROWTYPE;
    existing public.research_lab_official_baseline_unit_closures_v1%ROWTYPE;
    frontier_doc JSONB;
    closure_doc_value JSONB;
    closure_hash TEXT;
    closure_ref_value TEXT;
    attempt_count BIGINT;
    terminal_count BIGINT;
    uncertain_count BIGINT;
    inserted BOOLEAN := FALSE;
BEGIN
    IF NOT public.research_lab_official_baseline_exact_keys_v1(
        p_completion,
        ARRAY[
            'schema_version', 'run_sha256', 'unit_ref',
            'protocol_generation_sha256', 'raw_input_sha256',
            'start_request_sha256', 'terminal_result_sha256',
            'model_receipt_sha256', 'projection_sha256'
        ]
    )
       OR p_completion->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.official_baseline_unit_completion.v1'
       OR p_completion->>'run_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_completion->>'unit_ref' !~ '^baseline_icp:[0-9a-f]{64}$'
       OR p_completion->>'protocol_generation_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_completion->>'raw_input_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_completion->>'start_request_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_completion->>'terminal_result_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_completion->>'model_receipt_sha256' !~ '^sha256:[0-9a-f]{64}$'
       OR p_completion->>'projection_sha256' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_completion_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_official_baseline_reject_secret_doc_v1(
        p_completion, 'unit_completion'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            (p_completion->>'run_sha256') || ':' ||
                (p_completion->>'unit_ref'),
            0
        )
    );

    SELECT * INTO registered
      FROM public.research_lab_official_baseline_runs_v1 run
     WHERE run.run_sha256 = p_completion->>'run_sha256';
    IF NOT FOUND
       OR registered.protocol_generation_sha256 IS DISTINCT FROM
            p_completion->>'protocol_generation_sha256'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_release_conflict'
            USING ERRCODE = '23503';
    END IF;

    SELECT COUNT(*), COUNT(terminal.attempt_key),
           COUNT(*) FILTER (WHERE terminal.terminal_state = 'terminal_uncertain')
      INTO attempt_count, terminal_count, uncertain_count
      FROM public.research_lab_official_baseline_action_attempts_v1 attempt
      LEFT JOIN public.research_lab_official_baseline_action_terminals_v1 terminal
        ON terminal.attempt_key = attempt.attempt_key
     WHERE attempt.run_sha256 = p_completion->>'run_sha256'
       AND attempt.unit_ref = p_completion->>'unit_ref';
    IF attempt_count < 1
       OR terminal_count <> attempt_count
       OR uncertain_count <> 0
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_actions_incomplete'
            USING ERRCODE = '23514';
    END IF;

    frontier_doc := public.research_lab_official_baseline_provider_frontier_doc_v1(
        p_completion->>'run_sha256', p_completion->>'unit_ref'
    );
    closure_doc_value := pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_unit_closure.v1',
        'run_sha256', p_completion->>'run_sha256',
        'unit_ref', p_completion->>'unit_ref',
        'protocol_generation_sha256',
            p_completion->>'protocol_generation_sha256',
        'raw_input_sha256', p_completion->>'raw_input_sha256',
        'start_request_sha256', p_completion->>'start_request_sha256',
        'terminal_result_sha256', p_completion->>'terminal_result_sha256',
        'model_receipt_sha256', p_completion->>'model_receipt_sha256',
        'projection_sha256', p_completion->>'projection_sha256',
        'ordered_attempt_keys', frontier_doc->'ordered_attempt_keys',
        'ordered_attempt_sha256s', frontier_doc->'ordered_attempt_sha256s',
        'provider_frontier_sha256',
            public.research_lab_official_baseline_hash_v1(frontier_doc)
    );
    closure_hash :=
        public.research_lab_official_baseline_hash_v1(closure_doc_value);
    closure_ref_value := 'baseline_closure:' ||
        pg_catalog.substr(closure_hash, 8);

    INSERT INTO public.research_lab_official_baseline_unit_closures_v1 (
        closure_ref, closure_sha256, run_sha256, unit_ref,
        protocol_generation_sha256, raw_input_sha256, start_request_sha256,
        terminal_result_sha256, model_receipt_sha256, projection_sha256,
        ordered_attempt_keys, ordered_attempt_sha256s,
        provider_frontier_sha256, closure_doc
    ) VALUES (
        closure_ref_value, closure_hash, p_completion->>'run_sha256',
        p_completion->>'unit_ref', p_completion->>'protocol_generation_sha256',
        p_completion->>'raw_input_sha256',
        p_completion->>'start_request_sha256',
        p_completion->>'terminal_result_sha256',
        p_completion->>'model_receipt_sha256',
        p_completion->>'projection_sha256',
        frontier_doc->'ordered_attempt_keys',
        frontier_doc->'ordered_attempt_sha256s',
        public.research_lab_official_baseline_hash_v1(frontier_doc),
        closure_doc_value
    ) ON CONFLICT DO NOTHING
    RETURNING * INTO existing;
    inserted := FOUND;
    IF NOT inserted THEN
        SELECT * INTO existing
          FROM public.research_lab_official_baseline_unit_closures_v1 closure
         WHERE closure.run_sha256 = p_completion->>'run_sha256'
           AND closure.unit_ref = p_completion->>'unit_ref';
        IF NOT FOUND OR existing.closure_doc IS DISTINCT FROM closure_doc_value
        THEN
            RAISE EXCEPTION 'research_lab_official_baseline_unit_closure_conflict'
                USING ERRCODE = '23505';
        END IF;
    END IF;
    RETURN existing.closure_doc || pg_catalog.jsonb_build_object(
        'closure_ref', existing.closure_ref,
        'closure_sha256', existing.closure_sha256,
        'idempotent', NOT inserted
    );
END;
$close_unit$;

CREATE OR REPLACE FUNCTION public.research_lab_official_baseline_load_frontier_v1(
    p_run_sha256 TEXT,
    p_unit_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $load_frontier$
DECLARE
    existing public.research_lab_official_baseline_unit_closures_v1%ROWTYPE;
BEGIN
    IF p_run_sha256 !~ '^sha256:[0-9a-f]{64}$'
       OR p_unit_ref !~ '^baseline_icp:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_official_baseline_frontier_identity_invalid'
            USING ERRCODE = '22023';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_official_baseline_unit_closures_v1 closure
     WHERE closure.run_sha256 = p_run_sha256
       AND closure.unit_ref = p_unit_ref;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_official_baseline_unit_closure_missing'
            USING ERRCODE = '23503';
    END IF;
    RETURN existing.closure_doc || pg_catalog.jsonb_build_object(
        'closure_ref', existing.closure_ref,
        'closure_sha256', existing.closure_sha256,
        'idempotent', TRUE
    );
END;
$load_frontier$;

ALTER TABLE public.research_lab_official_baseline_runs_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_runs_v1
    FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_action_attempts_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_action_attempts_v1
    FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_action_terminals_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_action_terminals_v1
    FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_unit_closures_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_official_baseline_unit_closures_v1
    FORCE ROW LEVEL SECURITY;

DO $baseline_read_policies$
DECLARE
    relation_name TEXT;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'research_lab_official_baseline_runs_v1',
        'research_lab_official_baseline_action_attempts_v1',
        'research_lab_official_baseline_action_terminals_v1',
        'research_lab_official_baseline_unit_closures_v1'
    ] LOOP
        IF NOT EXISTS (
            SELECT 1
              FROM pg_catalog.pg_policies
             WHERE schemaname = 'public'
               AND tablename = relation_name
               AND policyname = 'service_role_read'
        ) THEN
            EXECUTE pg_catalog.format(
                'CREATE POLICY service_role_read ON public.%I '
                'FOR SELECT TO service_role USING (true)',
                relation_name
            );
        END IF;
    END LOOP;
END;
$baseline_read_policies$;

REVOKE ALL ON TABLE
    public.research_lab_official_baseline_runs_v1,
    public.research_lab_official_baseline_action_attempts_v1,
    public.research_lab_official_baseline_action_terminals_v1,
    public.research_lab_official_baseline_unit_closures_v1
FROM PUBLIC, anon, authenticated, service_role;
REVOKE TRUNCATE ON TABLE
    public.research_lab_official_baseline_runs_v1,
    public.research_lab_official_baseline_action_attempts_v1,
    public.research_lab_official_baseline_action_terminals_v1,
    public.research_lab_official_baseline_unit_closures_v1
FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE
    public.research_lab_official_baseline_runs_v1,
    public.research_lab_official_baseline_action_attempts_v1,
    public.research_lab_official_baseline_action_terminals_v1,
    public.research_lab_official_baseline_unit_closures_v1
TO service_role;

REVOKE ALL ON FUNCTION
    public.prevent_research_lab_official_baseline_mutation_v1()
FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_official_baseline_exact_keys_v1(JSONB, TEXT[]),
    public.research_lab_official_baseline_hash_v1(JSONB),
    public.research_lab_official_baseline_reject_secret_doc_v1(JSONB, TEXT),
    public.research_lab_official_baseline_provider_frontier_doc_v1(TEXT, TEXT)
FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_official_baseline_register_run_v1(JSONB),
    public.research_lab_official_baseline_reserve_action_v1(JSONB),
    public.research_lab_official_baseline_record_terminal_known_v1(JSONB),
    public.research_lab_official_baseline_record_terminal_uncertain_v1(JSONB),
    public.research_lab_official_baseline_load_replay_v1(JSONB),
    public.research_lab_official_baseline_close_unit_v1(JSONB),
    public.research_lab_official_baseline_load_frontier_v1(TEXT, TEXT)
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_official_baseline_register_run_v1(JSONB),
    public.research_lab_official_baseline_reserve_action_v1(JSONB),
    public.research_lab_official_baseline_record_terminal_known_v1(JSONB),
    public.research_lab_official_baseline_record_terminal_uncertain_v1(JSONB),
    public.research_lab_official_baseline_load_replay_v1(JSONB),
    public.research_lab_official_baseline_close_unit_v1(JSONB),
    public.research_lab_official_baseline_load_frontier_v1(TEXT, TEXT)
TO service_role;

COMMIT;
