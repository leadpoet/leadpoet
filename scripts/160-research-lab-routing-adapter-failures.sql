-- Durable, typed evidence for a provider adapter failure proven to occur
-- before the provider dispatch boundary.
--
-- This is intentionally separate from provider_attempts_v2.  A provider
-- attempt requires the complete V3 signed authorization/protected-release/
-- terminal graph.  A pre-dispatch adapter failure has no such graph and must
-- never be allowed to masquerade as one.  Promotion and provider billing
-- reconciliation read provider_attempts_v2 only, so this table cannot supply
-- provider-success evidence.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS public.research_lab_routing_adapter_failures_v2 (
    failure_key TEXT PRIMARY KEY
        CHECK (failure_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    provider_receipt_ref TEXT NOT NULL UNIQUE
        CHECK (provider_receipt_ref ~ '^provider_receipt:[0-9a-f]{16}$'),
    binding_id TEXT NOT NULL
        CHECK (binding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    tool_id TEXT NOT NULL
        CHECK (tool_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    variant_id TEXT NOT NULL
        CHECK (variant_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    unit_ref TEXT NOT NULL
        CHECK (unit_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    claim_key TEXT NOT NULL
        CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    request_fingerprint TEXT NOT NULL
        CHECK (request_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    outcome TEXT NOT NULL DEFAULT 'adapter_failure'
        CHECK (outcome = 'adapter_failure'),
    credit_microunits BIGINT NOT NULL DEFAULT 0
        CHECK (credit_microunits = 0),
    latency_ms BIGINT NOT NULL CHECK (latency_ms >= 0 AND latency_ms <= 900000),
    execution_mode TEXT NOT NULL
        CHECK (execution_mode IN ('fixture', 'replay', 'measured_lab')),
    failure_doc JSONB NOT NULL
        CHECK (pg_catalog.jsonb_typeof(failure_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE INDEX IF NOT EXISTS rl_route_adapter_failure_experiment_idx
    ON public.research_lab_routing_adapter_failures_v2(experiment_hash, failure_key);

DROP TRIGGER IF EXISTS research_lab_routing_adapter_failures_v2_append_only
    ON public.research_lab_routing_adapter_failures_v2;
CREATE TRIGGER research_lab_routing_adapter_failures_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_adapter_failures_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_adapter_failure_v3(
    p_failure_key TEXT,
    p_experiment_hash TEXT,
    p_provider_receipt_ref TEXT,
    p_binding_id TEXT,
    p_tool_id TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_request_fingerprint TEXT,
    p_latency_ms BIGINT,
    p_execution_mode TEXT,
    p_failure_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $append_adapter_failure_v3$
DECLARE
    existing public.research_lab_routing_adapter_failures_v2%ROWTYPE;
    failure_identity_lock BIGINT;
    receipt_identity_lock BIGINT;
BEGIN
    IF p_failure_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_provider_receipt_ref !~ '^provider_receipt:[0-9a-f]{16}$'
       OR p_binding_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_tool_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_unit_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR p_request_fingerprint !~ '^sha256:[0-9a-f]{64}$'
       OR p_latency_ms IS NULL OR p_latency_ms < 0 OR p_latency_ms > 900000
       OR p_execution_mode NOT IN ('fixture', 'replay', 'measured_lab')
       OR pg_catalog.jsonb_typeof(p_failure_doc) IS DISTINCT FROM 'object'
       OR p_failure_doc->>'schema_version'
            IS DISTINCT FROM 'leadpoet.research_lab.routing_adapter_failure.v3'
       OR p_failure_doc->>'failure_key' IS DISTINCT FROM p_failure_key
       OR p_failure_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR p_failure_doc->>'binding_id' IS DISTINCT FROM p_binding_id
       OR p_failure_doc->>'tool_id' IS DISTINCT FROM p_tool_id
       OR p_failure_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR p_failure_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR p_failure_doc->>'claim_key' IS DISTINCT FROM p_claim_key
       OR p_failure_doc->>'claim_generation'
            IS DISTINCT FROM p_claim_generation::TEXT
       OR p_failure_doc->>'request_fingerprint'
            IS DISTINCT FROM p_request_fingerprint
       OR p_failure_doc->>'outcome' IS DISTINCT FROM 'adapter_failure'
       OR p_failure_doc->>'credit_microunits' IS DISTINCT FROM '0'
       OR p_failure_doc->>'latency_ms' IS DISTINCT FROM p_latency_ms::TEXT
       OR p_failure_doc->>'execution_mode' IS DISTINCT FROM p_execution_mode
       OR p_failure_doc->>'pre_dispatch' IS DISTINCT FROM 'true'
       OR pg_catalog.jsonb_typeof(p_failure_doc->'provider_receipt')
            IS DISTINCT FROM 'object'
       OR p_failure_key IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'contract_version', 'leadpoet.provider_receipt_key:v2',
                'tool_id', p_failure_doc->'provider_receipt'->>'tool_id',
                'binding_version',
                    p_failure_doc->'provider_receipt'->>'binding_version',
                'request_fingerprint',
                    p_failure_doc->'provider_receipt'->>'request_fingerprint'
            )
          )
       OR ARRAY(
            SELECT key
              FROM pg_catalog.jsonb_object_keys(p_failure_doc) AS keys(key)
             ORDER BY key
          ) IS DISTINCT FROM ARRAY[
            'binding_id', 'claim_generation', 'claim_key', 'credit_microunits',
            'execution_mode', 'experiment_hash', 'failure_key', 'latency_ms',
            'outcome', 'pre_dispatch', 'provider_receipt',
            'request_fingerprint', 'schema_version', 'tool_id', 'unit_ref',
            'variant_id'
          ]::TEXT[]
       OR ARRAY(
            SELECT key
              FROM pg_catalog.jsonb_object_keys(p_failure_doc->'provider_receipt')
                    AS keys(key)
             ORDER BY key
          ) IS DISTINCT FROM ARRAY[
            'binding_id', 'binding_version', 'credit_microunits',
            'evidence_hash', 'execution_mode', 'latency_ms', 'outcome',
            'receipt_ref', 'request_fingerprint', 'source_lineage_id',
            'tool_id', 'unit_ref'
          ]::TEXT[]
       OR p_failure_doc->'provider_receipt'->>'receipt_ref'
            IS DISTINCT FROM p_provider_receipt_ref
       OR p_provider_receipt_ref IS DISTINCT FROM (
            'provider_receipt:' || pg_catalog.substr(
                public.research_lab_routing_jsonb_hash_v2(
                    (p_failure_doc->'provider_receipt') - 'receipt_ref'::TEXT
                ), 8, 16
            )
          )
       OR p_failure_doc->'provider_receipt'->>'binding_id'
            IS DISTINCT FROM p_binding_id
       OR p_failure_doc->'provider_receipt'->>'tool_id'
            IS DISTINCT FROM p_tool_id
       OR p_failure_doc->'provider_receipt'->>'unit_ref'
            IS DISTINCT FROM p_unit_ref
       OR p_failure_doc->'provider_receipt'->>'request_fingerprint'
            IS DISTINCT FROM p_request_fingerprint
       OR p_failure_doc->'provider_receipt'->>'outcome'
            IS DISTINCT FROM 'adapter_failure'
       OR p_failure_doc->'provider_receipt'->>'credit_microunits'
            IS DISTINCT FROM '0'
       OR p_failure_doc->'provider_receipt'->>'latency_ms'
            IS DISTINCT FROM p_latency_ms::TEXT
       OR p_failure_doc->'provider_receipt'->>'execution_mode'
            IS DISTINCT FROM p_execution_mode
       OR p_failure_doc->'provider_receipt'->>'binding_version'
            !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_failure_doc->'provider_receipt'->>'source_lineage_id'
            !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_failure_doc->'provider_receipt'->>'evidence_hash'
            !~ '^sha256:[0-9a-f]{64}$'
       OR p_failure_doc->'provider_receipt'->>'receipt_ref'
            !~ '^provider_receipt:[0-9a-f]{16}$'
       OR p_failure_doc ?| ARRAY[
            'terminal_proof', 'terminal_result', 'terminal_execution_receipt',
            'protected_release_receipt', 'call_grant_receipt', 'admission_bundle'
       ]
    THEN
        RAISE EXCEPTION 'research_lab_routing_adapter_failure_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;

    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_failure_doc, 'routing adapter failure v3'
    );

    -- Exact replay is safe without a live claim.  It only returns the
    -- original row; a changed identity or document is a conflict.
    -- Use the same deterministic two-identity lock order as the provider
    -- attempt V3 RPC.  The ledgers are separate tables, so their local unique
    -- indexes cannot serialize a shared key or provider receipt reference.
    failure_identity_lock := pg_catalog.hashtextextended(p_failure_key, 0);
    receipt_identity_lock := pg_catalog.hashtextextended(p_provider_receipt_ref, 0);
    IF failure_identity_lock <= receipt_identity_lock THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(failure_identity_lock);
        IF receipt_identity_lock <> failure_identity_lock THEN
            PERFORM pg_catalog.pg_advisory_xact_lock(receipt_identity_lock);
        END IF;
    ELSE
        PERFORM pg_catalog.pg_advisory_xact_lock(receipt_identity_lock);
        PERFORM pg_catalog.pg_advisory_xact_lock(failure_identity_lock);
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_routing_adapter_failures_v2
     WHERE failure_key = p_failure_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.provider_receipt_ref IS DISTINCT FROM p_provider_receipt_ref
           OR existing.binding_id IS DISTINCT FROM p_binding_id
           OR existing.tool_id IS DISTINCT FROM p_tool_id
           OR existing.variant_id IS DISTINCT FROM p_variant_id
           OR existing.unit_ref IS DISTINCT FROM p_unit_ref
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.request_fingerprint IS DISTINCT FROM p_request_fingerprint
           OR existing.latency_ms IS DISTINCT FROM p_latency_ms
           OR existing.execution_mode IS DISTINCT FROM p_execution_mode
           OR existing.failure_doc IS DISTINCT FROM p_failure_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_adapter_failure_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'failure_key', p_failure_key, 'idempotent', TRUE
        );
    END IF;

    -- The same content-addressed key or receipt ref must never be dual-use
    -- across the typed failure and provider-attempt ledgers.
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_routing_provider_attempts_v2 attempt
         WHERE attempt.attempt_key = p_failure_key
            OR attempt.provider_receipt_ref = p_provider_receipt_ref
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_adapter_failure_v3_provider_attempt_collision'
            USING ERRCODE = '23505';
    END IF;

    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );

    -- The model-owned binding and variant must be the exact pair submitted
    -- for this immutable experiment.  Unit refs remain opaque but are bound
    -- into the receipt identity and failure document above.
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiments_v2 experiment
          CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
              experiment.spec_doc->'provider_bindings'
          ) binding(value)
          CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
              experiment.spec_doc->'variants'
          ) variant(value)
         WHERE experiment.experiment_hash = p_experiment_hash
           AND experiment.receipt_execution_mode = p_execution_mode
           AND binding.value->>'binding_id' = p_binding_id
           AND binding.value->>'tool_id' = p_tool_id
           AND variant.value->>'variant_id' = p_variant_id
           AND p_binding_id = ANY (
               ARRAY(
                   SELECT pg_catalog.jsonb_array_elements_text(
                       variant.value->'binding_ids'
                   )
               )
           )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_adapter_failure_v3_binding_not_declared'
            USING ERRCODE = '23503';
    END IF;

    -- A pre-dispatch failure is valid only while no durable dispatch marker
    -- exists for this exact request.  If the boundary was crossed, the
    -- caller must preserve the reservation as uncertain and obtain terminal
    -- provider evidence; this RPC cannot erase that ambiguity.
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.experiment_hash = p_experiment_hash
           AND event.claim_key = p_claim_key
           AND event.claim_generation = p_claim_generation
           AND event.event_type = 'provider_dispatch_started'
           AND event.event_doc->>'request_fingerprint' = p_request_fingerprint
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_adapter_failure_v3_dispatch_started'
            USING ERRCODE = '23503';
    END IF;

    INSERT INTO public.research_lab_routing_adapter_failures_v2 (
        failure_key, experiment_hash, provider_receipt_ref, binding_id, tool_id,
        variant_id, unit_ref, claim_key, claim_generation, request_fingerprint,
        outcome, credit_microunits, latency_ms, execution_mode, failure_doc
    ) VALUES (
        p_failure_key, p_experiment_hash, p_provider_receipt_ref, p_binding_id,
        p_tool_id, p_variant_id, p_unit_ref, p_claim_key, p_claim_generation,
        p_request_fingerprint, 'adapter_failure', 0, p_latency_ms,
        p_execution_mode, p_failure_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'failure_key', p_failure_key, 'idempotent', FALSE
    );
END;
$append_adapter_failure_v3$;

ALTER TABLE public.research_lab_routing_adapter_failures_v2 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.research_lab_routing_adapter_failures_v2
    FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_routing_adapter_failures_v2 TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_adapter_failure_v3(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, BIGINT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_adapter_failure_v3(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, BIGINT, TEXT, JSONB
) TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;
