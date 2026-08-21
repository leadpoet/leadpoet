-- Durable, append-only authority for measured Research Lab routing experiments.
--
-- This migration deliberately stores only redacted hashes, ids, bounded
-- outcome metadata, and immutable receipts.  It never stores a provider
-- request, response body, credential, or authorization header.  Every write
-- is through a fixed-search-path SECURITY DEFINER RPC and is idempotent only
-- when the full submitted record is byte-for-byte equal to the original.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Python routing identities use RFC-8259 JSON without whitespace and with
-- bytewise-sorted object keys. PostgreSQL JSONB text is deterministic but
-- includes different whitespace, so the database must rebuild the same
-- canonical bytes before it accepts a caller-supplied content hash.
CREATE OR REPLACE FUNCTION public.research_lab_routing_canonical_jsonb_v2(
    p_value JSONB
)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $canonical_json$
BEGIN
    CASE pg_catalog.jsonb_typeof(p_value)
        WHEN 'object' THEN
            RETURN (
                SELECT '{' || coalesce(
                    pg_catalog.string_agg(
                        pg_catalog.to_jsonb(entry.key)::TEXT || ':' ||
                        public.research_lab_routing_canonical_jsonb_v2(entry.value),
                        ',' ORDER BY entry.key COLLATE "C"
                    ), ''
                ) || '}'
                FROM pg_catalog.jsonb_each(p_value) AS entry(key, value)
            );
        WHEN 'array' THEN
            RETURN (
                SELECT '[' || coalesce(
                    pg_catalog.string_agg(
                        public.research_lab_routing_canonical_jsonb_v2(entry.value),
                        ',' ORDER BY entry.ordinality
                    ), ''
                ) || ']'
                FROM pg_catalog.jsonb_array_elements(p_value)
                    WITH ORDINALITY AS entry(value, ordinality)
            );
        ELSE
            RETURN p_value::TEXT;
    END CASE;
END;
$canonical_json$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_jsonb_hash_v2(
    p_value JSONB
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $jsonb_hash$
    SELECT 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                public.research_lab_routing_canonical_jsonb_v2(p_value),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    )
$jsonb_hash$;

CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiments_v2 (
    experiment_hash TEXT PRIMARY KEY
        CHECK (experiment_hash ~ '^sha256:[0-9a-f]{64}$'),
    experiment_id TEXT NOT NULL CHECK (experiment_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    spec_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(spec_doc) = 'object'),
    receipt_execution_mode TEXT NOT NULL
        CHECK (receipt_execution_mode IN ('fixture', 'replay', 'measured_lab')),
    allow_live_credit_spend BOOLEAN NOT NULL,
    execution_envelope_hash TEXT
        CHECK (execution_envelope_hash IS NULL OR execution_envelope_hash ~ '^sha256:[0-9a-f]{64}$'),
    execution_envelope_doc JSONB
        CHECK (execution_envelope_doc IS NULL OR pg_catalog.jsonb_typeof(execution_envelope_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (spec_doc ? 'contract_version'),
    CHECK (spec_doc ? 'credit_budget'),
    CHECK ((execution_envelope_hash IS NULL) = (execution_envelope_doc IS NULL)),
    CHECK (NOT allow_live_credit_spend OR execution_envelope_hash IS NOT NULL)
);
ALTER TABLE public.research_lab_routing_experiments_v2
    ADD COLUMN IF NOT EXISTS execution_envelope_hash TEXT
        CHECK (execution_envelope_hash IS NULL OR execution_envelope_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS execution_envelope_doc JSONB
        CHECK (execution_envelope_doc IS NULL OR pg_catalog.jsonb_typeof(execution_envelope_doc) = 'object');
ALTER TABLE public.research_lab_routing_experiments_v2
    DROP CONSTRAINT IF EXISTS research_lab_routing_experiment_execution_envelope_pair_v2;
ALTER TABLE public.research_lab_routing_experiments_v2
    ADD CONSTRAINT research_lab_routing_experiment_execution_envelope_pair_v2
    CHECK ((execution_envelope_hash IS NULL) = (execution_envelope_doc IS NULL));
ALTER TABLE public.research_lab_routing_experiments_v2
    DROP CONSTRAINT IF EXISTS research_lab_routing_experiment_live_envelope_v2;
ALTER TABLE public.research_lab_routing_experiments_v2
    ADD CONSTRAINT research_lab_routing_experiment_live_envelope_v2
    CHECK (NOT allow_live_credit_spend OR execution_envelope_hash IS NOT NULL);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_events_v2 (
    event_hash TEXT PRIMARY KEY CHECK (event_hash ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    event_type TEXT NOT NULL CHECK (event_type IN (
        'submitted', 'claimed', 'claim_recovered', 'run_started', 'run_completed',
        'run_failed', 'promotion_requested', 'promoted'
    )),
    event_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(event_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
-- Keep the event vocabulary explicit even if this migration is replayed
-- after a partially applied development run.  A durable pre-provider
-- dispatch marker is the only proof that lets crash recovery distinguish a
-- safe pre-dispatch release from an unknown post-dispatch charge.
ALTER TABLE public.research_lab_routing_experiment_events_v2
    DROP CONSTRAINT IF EXISTS research_lab_routing_experiment_events_v2_event_type_check;
ALTER TABLE public.research_lab_routing_experiment_events_v2
    ADD CONSTRAINT research_lab_routing_experiment_events_v2_event_type_check
    CHECK (event_type IN (
        'submitted', 'claimed', 'claim_recovered', 'run_started', 'run_completed',
        'run_failed', 'promotion_requested', 'promoted', 'provider_dispatch_started'
    ));
-- Internal submit/claim/recovery/promotion events predate an execution
-- claim. Every worker-execution event has both fields populated and is
-- checked against the current fencing generation in its RPC.
ALTER TABLE public.research_lab_routing_experiment_events_v2
    ADD COLUMN IF NOT EXISTS claim_key TEXT
        CHECK (claim_key IS NULL OR claim_key ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS claim_generation BIGINT
        CHECK (claim_generation IS NULL OR claim_generation > 0);
CREATE INDEX IF NOT EXISTS rl_route_event_exp_created_idx
    ON public.research_lab_routing_experiment_events_v2(experiment_hash, created_at);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_execution_requests_v2 (
    request_hash TEXT PRIMARY KEY CHECK (request_hash ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL UNIQUE
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    request_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(request_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
CREATE INDEX IF NOT EXISTS rl_route_request_created_idx
    ON public.research_lab_routing_execution_requests_v2(created_at, experiment_hash);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_claims_v2 (
    claim_key TEXT PRIMARY KEY CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    -- The claim is fenced by this immutable identity and the database lease.
    -- V3 does not persist a bearer capability.  This nullable legacy field is
    -- retained only for old rows and revoked v2 replay compatibility.
    claim_capability_commitment TEXT CHECK (claim_capability_commitment IS NULL OR claim_capability_commitment ~ '^sha256:[0-9a-f]{64}$'),
    request_hash TEXT CHECK (request_hash IS NULL OR request_hash ~ '^sha256:[0-9a-f]{64}$'),
    lease_hash TEXT CHECK (lease_hash IS NULL OR lease_hash ~ '^sha256:[0-9a-f]{64}$'),
    lease_generation BIGINT CHECK (lease_generation IS NULL OR lease_generation > 0),
    worker_ref TEXT NOT NULL CHECK (worker_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    claim_state TEXT NOT NULL CHECK (claim_state IN ('claimed', 'recovered')),
    lease_expires_at TIMESTAMPTZ,
    claim_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(claim_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK ((request_hash IS NULL AND lease_hash IS NULL AND lease_generation IS NULL)
        OR (request_hash IS NOT NULL AND lease_hash IS NOT NULL AND lease_generation IS NOT NULL))
);
ALTER TABLE public.research_lab_routing_experiment_claims_v2
    ADD COLUMN IF NOT EXISTS claim_capability_commitment TEXT
        CHECK (claim_capability_commitment IS NULL OR claim_capability_commitment ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS request_hash TEXT
        CHECK (request_hash IS NULL OR request_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS lease_hash TEXT
        CHECK (lease_hash IS NULL OR lease_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS lease_generation BIGINT
        CHECK (lease_generation IS NULL OR lease_generation > 0);
ALTER TABLE public.research_lab_routing_experiment_claims_v2
    DROP CONSTRAINT IF EXISTS research_lab_routing_claim_execution_lease_pair_v3;
ALTER TABLE public.research_lab_routing_experiment_claims_v2
    ADD CONSTRAINT research_lab_routing_claim_execution_lease_pair_v3
    CHECK ((request_hash IS NULL AND lease_hash IS NULL AND lease_generation IS NULL)
        OR (request_hash IS NOT NULL AND lease_hash IS NOT NULL AND lease_generation IS NOT NULL));
CREATE UNIQUE INDEX IF NOT EXISTS rl_route_claim_generation_uq
    ON public.research_lab_routing_experiment_claims_v2(experiment_hash, claim_generation);
CREATE INDEX IF NOT EXISTS rl_route_claim_head_idx
    ON public.research_lab_routing_experiment_claims_v2(experiment_hash, created_at DESC);

-- Lease changes and terminal closure are append-only facts.  They are kept
-- outside the immutable claim row so a renew/heartbeat cannot turn an UPDATE
-- into an authority escape hatch.
CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_claim_heartbeats_v2 (
    heartbeat_key TEXT PRIMARY KEY CHECK (heartbeat_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    claim_capability_commitment TEXT CHECK (claim_capability_commitment IS NULL OR claim_capability_commitment ~ '^sha256:[0-9a-f]{64}$'),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    heartbeat_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(heartbeat_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.research_lab_routing_experiment_claim_heartbeats_v2
    ADD COLUMN IF NOT EXISTS claim_capability_commitment TEXT
        CHECK (claim_capability_commitment IS NULL OR claim_capability_commitment ~ '^sha256:[0-9a-f]{64}$');
CREATE INDEX IF NOT EXISTS rl_route_claim_heartbeat_head_idx
    ON public.research_lab_routing_experiment_claim_heartbeats_v2(
        experiment_hash, claim_key, claim_generation, created_at DESC
    );

CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_claim_closures_v2 (
    close_key TEXT PRIMARY KEY CHECK (close_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    claim_capability_commitment TEXT CHECK (claim_capability_commitment IS NULL OR claim_capability_commitment ~ '^sha256:[0-9a-f]{64}$'),
    close_reason TEXT NOT NULL CHECK (close_reason IN ('completed', 'failed', 'cancelled')),
    close_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(close_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.research_lab_routing_experiment_claim_closures_v2
    ADD COLUMN IF NOT EXISTS claim_capability_commitment TEXT
        CHECK (claim_capability_commitment IS NULL OR claim_capability_commitment ~ '^sha256:[0-9a-f]{64}$');
CREATE UNIQUE INDEX IF NOT EXISTS rl_route_claim_close_once_uq
    ON public.research_lab_routing_experiment_claim_closures_v2(
        experiment_hash, claim_key, claim_generation
    );

CREATE TABLE IF NOT EXISTS public.research_lab_routing_provider_attempts_v2 (
    attempt_key TEXT PRIMARY KEY CHECK (attempt_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    provider_receipt_ref TEXT NOT NULL UNIQUE
        CHECK (provider_receipt_ref ~ '^provider_receipt:[0-9a-f]{16}$'),
    binding_id TEXT NOT NULL CHECK (binding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    tool_id TEXT NOT NULL CHECK (tool_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    variant_id TEXT NOT NULL CHECK (variant_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    unit_ref TEXT NOT NULL CHECK (unit_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    reservation_id TEXT NOT NULL CHECK (reservation_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    action_id TEXT NOT NULL CHECK (action_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    binding_catalog_manifest_hash TEXT NOT NULL
        CHECK (binding_catalog_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    authorization_hash TEXT NOT NULL
        CHECK (authorization_hash ~ '^sha256:[0-9a-f]{64}$'),
    authorization_proof_hash TEXT NOT NULL
        CHECK (authorization_proof_hash ~ '^sha256:[0-9a-f]{64}$'),
    request_body_hash TEXT NOT NULL
        CHECK (request_body_hash ~ '^sha256:[0-9a-f]{64}$'),
    -- These hashes are redundant by design.  They make the database receipt
    -- chain independently auditable without trusting a Python projection.
    terminal_receipt_hash TEXT
        CHECK (terminal_receipt_hash IS NULL OR terminal_receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    protected_release_receipt_hash TEXT
        CHECK (protected_release_receipt_hash IS NULL OR protected_release_receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    admission_bundle_hash TEXT
        CHECK (admission_bundle_hash IS NULL OR admission_bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    terminal_provider_record_hash TEXT
        CHECK (terminal_provider_record_hash IS NULL OR terminal_provider_record_hash ~ '^sha256:[0-9a-f]{64}$'),
    terminal_billing_projection_hash TEXT
        CHECK (terminal_billing_projection_hash IS NULL OR terminal_billing_projection_hash ~ '^sha256:[0-9a-f]{64}$'),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    request_fingerprint TEXT NOT NULL CHECK (request_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    outcome TEXT NOT NULL CHECK (outcome IN ('verified', 'rejected', 'source_miss', 'adapter_failure')),
    credit_microunits BIGINT NOT NULL CHECK (credit_microunits >= 0),
    billing_state TEXT NOT NULL CHECK (billing_state IN ('known', 'uncertain')),
    authoritative_billed_credit_microunits BIGINT
        CHECK (authoritative_billed_credit_microunits >= 0),
    latency_ms BIGINT NOT NULL CHECK (latency_ms >= 0),
    execution_mode TEXT NOT NULL CHECK (execution_mode IN ('fixture', 'replay', 'measured_lab')),
    attempt_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(attempt_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK (credit_microunits <= 10000000),
    CHECK (latency_ms <= 900000),
    CHECK (
        (billing_state = 'known'
         AND authoritative_billed_credit_microunits IS NOT NULL)
        OR (billing_state = 'uncertain'
            AND authoritative_billed_credit_microunits IS NULL)
    ),
    -- A synthetic adapter failure remains a zero-cost receipt.  An
    -- authoritative broker can separately report a known billed amount, and
    -- an unknown charge is held by an uncertain budget reservation.
    CHECK (
        outcome <> 'adapter_failure'
        OR credit_microunits = 0
    )
);
ALTER TABLE public.research_lab_routing_provider_attempts_v2
    ADD COLUMN IF NOT EXISTS binding_catalog_manifest_hash TEXT
        CHECK (binding_catalog_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS authorization_hash TEXT
        CHECK (authorization_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS authorization_proof_hash TEXT
        CHECK (authorization_proof_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS request_body_hash TEXT
        CHECK (request_body_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS terminal_receipt_hash TEXT
        CHECK (terminal_receipt_hash IS NULL OR terminal_receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS protected_release_receipt_hash TEXT
        CHECK (protected_release_receipt_hash IS NULL OR protected_release_receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS admission_bundle_hash TEXT
        CHECK (admission_bundle_hash IS NULL OR admission_bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS terminal_provider_record_hash TEXT
        CHECK (terminal_provider_record_hash IS NULL OR terminal_provider_record_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS terminal_billing_projection_hash TEXT
        CHECK (terminal_billing_projection_hash IS NULL OR terminal_billing_projection_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS authorization_request_hash TEXT
        CHECK (authorization_request_hash IS NULL OR authorization_request_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS terminal_request_hash TEXT
        CHECK (terminal_request_hash IS NULL OR terminal_request_hash ~ '^sha256:[0-9a-f]{64}$'),
    ADD COLUMN IF NOT EXISTS terminal_result_hash TEXT
        CHECK (terminal_result_hash IS NULL OR terminal_result_hash ~ '^sha256:[0-9a-f]{64}$');

-- The five links below are part of the provider-attempt authority contract,
-- not optional metadata.  Keep these checks NOT VALID so a replay of an
-- earlier installation with legacy NULL rows remains possible, while every
-- new or changed attempt is rejected unless it carries the complete proof
-- chain.  A later data-repair migration may validate them after legacy rows
-- have been reconciled.
ALTER TABLE public.research_lab_routing_provider_attempts_v2
    DROP CONSTRAINT IF EXISTS rl_route_attempt_terminal_receipt_required,
    DROP CONSTRAINT IF EXISTS rl_route_attempt_protected_receipt_required,
    DROP CONSTRAINT IF EXISTS rl_route_attempt_admission_required,
    DROP CONSTRAINT IF EXISTS rl_route_attempt_provider_record_required,
    DROP CONSTRAINT IF EXISTS rl_route_attempt_billing_projection_required;
ALTER TABLE public.research_lab_routing_provider_attempts_v2
    ADD CONSTRAINT rl_route_attempt_terminal_receipt_required
        CHECK (terminal_receipt_hash IS NOT NULL) NOT VALID,
    ADD CONSTRAINT rl_route_attempt_protected_receipt_required
        CHECK (protected_release_receipt_hash IS NOT NULL) NOT VALID,
    ADD CONSTRAINT rl_route_attempt_admission_required
        CHECK (admission_bundle_hash IS NOT NULL) NOT VALID,
    ADD CONSTRAINT rl_route_attempt_provider_record_required
        CHECK (terminal_provider_record_hash IS NOT NULL) NOT VALID,
    ADD CONSTRAINT rl_route_attempt_billing_projection_required
        CHECK (terminal_billing_projection_hash IS NOT NULL) NOT VALID;
CREATE INDEX IF NOT EXISTS rl_route_attempt_experiment_idx
    ON public.research_lab_routing_provider_attempts_v2(experiment_hash, binding_id, created_at);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_decision_receipts_v2 (
    receipt_id TEXT PRIMARY KEY CHECK (receipt_id ~ '^routing_decision:[0-9a-f]{16}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    variant_id TEXT NOT NULL CHECK (variant_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    unit_ref TEXT NOT NULL CHECK (unit_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    plan_hash TEXT NOT NULL CHECK (plan_hash ~ '^sha256:[0-9a-f]{64}$'),
    route_hash TEXT NOT NULL CHECK (route_hash ~ '^sha256:[0-9a-f]{64}$'),
    decision_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(decision_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
CREATE INDEX IF NOT EXISTS rl_route_decision_experiment_idx
    ON public.research_lab_routing_decision_receipts_v2(experiment_hash, variant_id, unit_ref);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_evaluation_receipts_v2 (
    receipt_id TEXT PRIMARY KEY CHECK (receipt_id ~ '^routing_evaluation_v2:[0-9a-f]{16}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    evaluation_hash TEXT NOT NULL UNIQUE CHECK (evaluation_hash ~ '^sha256:[0-9a-f]{64}$'),
    selected_variant_id TEXT NOT NULL CHECK (selected_variant_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    evaluation_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(evaluation_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_lab_references_v2 (
    reference_hash TEXT PRIMARY KEY CHECK (reference_hash ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    evaluation_receipt_id TEXT NOT NULL REFERENCES public.research_lab_routing_evaluation_receipts_v2(receipt_id),
    selected_variant_id TEXT NOT NULL CHECK (selected_variant_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    reconciliation_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(reconciliation_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    UNIQUE (evaluation_receipt_id)
);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_budget_events_v2 (
    event_key TEXT PRIMARY KEY CHECK (event_key ~ '^sha256:[0-9a-f]{64}$'),
    reservation_id TEXT NOT NULL CHECK (reservation_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    experiment_hash TEXT NOT NULL REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    binding_id TEXT NOT NULL CHECK (binding_id ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    attempt_key TEXT CHECK (attempt_key ~ '^sha256:[0-9a-f]{64}$'),
    event_type TEXT NOT NULL CHECK (event_type IN ('reserve', 'settle', 'uncertain', 'recover')),
    credit_microunits BIGINT NOT NULL CHECK (credit_microunits >= 0),
    lease_expires_at TIMESTAMPTZ,
    event_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(event_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.research_lab_routing_budget_events_v2
    DROP CONSTRAINT IF EXISTS research_lab_routing_budget_events_v2_event_type_check;
ALTER TABLE public.research_lab_routing_budget_events_v2
    ADD CONSTRAINT research_lab_routing_budget_events_v2_event_type_check
    CHECK (event_type IN ('reserve', 'settle', 'uncertain', 'recover'));
-- The runtime derives a reservation identifier from the experiment, variant,
-- unit, tool, and attempt.  Make that identity globally single-use as well:
-- a service caller must not be able to make a same-named reservation in a
-- second experiment and confuse a later settle/recover RPC that accepts only
-- the reservation identity.
CREATE UNIQUE INDEX IF NOT EXISTS rl_route_reservation_global_uq
    ON public.research_lab_routing_budget_events_v2(reservation_id)
    WHERE event_type = 'reserve';
CREATE INDEX IF NOT EXISTS rl_route_budget_head_idx
    ON public.research_lab_routing_budget_events_v2(experiment_hash, reservation_id, created_at DESC);

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_only_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $append_only$
BEGIN
    RAISE EXCEPTION 'research_lab_routing_v2_append_only';
END;
$append_only$;

DROP TRIGGER IF EXISTS research_lab_routing_experiments_v2_append_only
    ON public.research_lab_routing_experiments_v2;
CREATE TRIGGER research_lab_routing_experiments_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_experiments_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_experiment_events_v2_append_only
    ON public.research_lab_routing_experiment_events_v2;
CREATE TRIGGER research_lab_routing_experiment_events_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_experiment_events_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_experiment_claims_v2_append_only
    ON public.research_lab_routing_experiment_claims_v2;
CREATE TRIGGER research_lab_routing_experiment_claims_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_experiment_claims_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_execution_requests_v2_append_only
    ON public.research_lab_routing_execution_requests_v2;
CREATE TRIGGER research_lab_routing_execution_requests_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_execution_requests_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_experiment_claim_heartbeats_v2_append_only
    ON public.research_lab_routing_experiment_claim_heartbeats_v2;
CREATE TRIGGER research_lab_routing_experiment_claim_heartbeats_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_experiment_claim_heartbeats_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_experiment_claim_closures_v2_append_only
    ON public.research_lab_routing_experiment_claim_closures_v2;
CREATE TRIGGER research_lab_routing_experiment_claim_closures_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_experiment_claim_closures_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_provider_attempts_v2_append_only
    ON public.research_lab_routing_provider_attempts_v2;
CREATE TRIGGER research_lab_routing_provider_attempts_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_provider_attempts_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_decision_receipts_v2_append_only
    ON public.research_lab_routing_decision_receipts_v2;
CREATE TRIGGER research_lab_routing_decision_receipts_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_decision_receipts_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_evaluation_receipts_v2_append_only
    ON public.research_lab_routing_evaluation_receipts_v2;
CREATE TRIGGER research_lab_routing_evaluation_receipts_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_evaluation_receipts_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_lab_references_v2_append_only
    ON public.research_lab_routing_lab_references_v2;
CREATE TRIGGER research_lab_routing_lab_references_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_lab_references_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();
DROP TRIGGER IF EXISTS research_lab_routing_budget_events_v2_append_only
    ON public.research_lab_routing_budget_events_v2;
CREATE TRIGGER research_lab_routing_budget_events_v2_append_only
    BEFORE UPDATE OR DELETE ON public.research_lab_routing_budget_events_v2
    FOR EACH ROW EXECUTE FUNCTION public.research_lab_routing_append_only_v2();

CREATE OR REPLACE FUNCTION public.research_lab_routing_reject_secret_doc_v2(
    p_doc JSONB,
    p_name TEXT
)
RETURNS VOID
LANGUAGE plpgsql
IMMUTABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $reject_secret$
DECLARE
    normalized TEXT := lower(coalesce(p_doc::TEXT, ''));
    -- Remove only a closed set of exact SHA-256 commitment fields from the
    -- key-name scan.  A field with a non-hash value remains in the scan and
    -- is rejected.  Raw values and every unlisted secret-shaped key retain
    -- the original broad rejection rule below.
    secret_scan_text TEXT;
    document_bytes INTEGER := pg_catalog.pg_column_size(p_doc);
    nesting_depth INTEGER;
BEGIN
    secret_scan_text := pg_catalog.regexp_replace(
        normalized,
        '"(authorization_hash|authorization_proof_hash|authorization_request_hash|request_body_hash|response_body_hash|claim_fence_hash)"[[:space:]]*:[[:space:]]*"sha256:[0-9a-f]{64}"',
        '',
        'g'
    );
    secret_scan_text := pg_catalog.regexp_replace(
        secret_scan_text,
        '"(authorization_job_id|job_id)"[[:space:]]*:[[:space:]]*"[a-z0-9][a-z0-9_.:/@+-]{0,191}"',
        '',
        'g'
    );
    WITH RECURSIVE tree(value, depth) AS (
        SELECT p_doc, 1
        UNION ALL
        SELECT child.value, tree.depth + 1
        FROM tree
        CROSS JOIN LATERAL (
            SELECT value FROM pg_catalog.jsonb_each(tree.value)
             WHERE pg_catalog.jsonb_typeof(tree.value) = 'object'
            UNION ALL
            SELECT value FROM pg_catalog.jsonb_array_elements(tree.value)
             WHERE pg_catalog.jsonb_typeof(tree.value) = 'array'
        ) AS child
        WHERE tree.depth < 33
    )
    SELECT max(depth) INTO nesting_depth FROM tree;
    IF pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
       OR document_bytes > 65536
       OR coalesce(nesting_depth, 0) > 32
       OR secret_scan_text ~ '(api[_-]?key|authorization|bearer[[:space:]]|access[_-]?token|secret[_-]?key|claim[_-]?(capability|nonce|fence)|raw[_-]?payload|response[_-]?body([^[:alnum:]_]|$)|request[_-]?body([^[:alnum:]_]|$)|response[_-]?text|password|passwd|private[_-]?key|client[_-]?secret|credential|service[_-]?role)'
       OR normalized ~ '(https?|postgres(ql)?|redis)://[^[:space:]"'']*:[^[:space:]"''@]*@'
    THEN
        RAISE EXCEPTION '% is invalid or contains forbidden material', p_name
            USING ERRCODE = '22023';
    END IF;
END;
$reject_secret$;

-- The worker holds one ephemeral, random SHA-256-shaped capability in memory.
-- Only a domain-separated commitment is persisted.  Hashing the already
-- random capability under a fixed domain makes a table-read commitment
-- unusable as the bearer for this function or any other hash namespace.
CREATE OR REPLACE FUNCTION public.research_lab_routing_claim_capability_commitment_v2(
    p_claim_nonce TEXT
)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
STRICT
SECURITY DEFINER
SET search_path = pg_catalog
AS $claim_commitment$
BEGIN
    IF p_claim_nonce !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_capability_invalid' USING ERRCODE = '22023';
    END IF;
    RETURN 'sha256:' || pg_catalog.encode(
        pg_catalog.sha256(
            pg_catalog.convert_to(
                'leadpoet.routing.claim-capability-commitment.v2:' || p_claim_nonce,
                'UTF8'
            )
        ),
        'hex'
    );
END;
$claim_commitment$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_claim_v2(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $assert_claim$
DECLARE
    head public.research_lab_routing_experiment_claims_v2%ROWTYPE;
    expected_commitment TEXT;
    effective_expiry TIMESTAMPTZ;
BEGIN
    expected_commitment := public.research_lab_routing_claim_capability_commitment_v2(p_claim_nonce);
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation < 1
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_fence_invalid' USING ERRCODE = '22023';
    END IF;
    -- Fencing and every subsequent caller mutation share the experiment
    -- transaction lock. A stale worker cannot validate generation N, pause,
    -- and then append after recovery created generation N+1.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    SELECT * INTO head
      FROM public.research_lab_routing_experiment_claims_v2
     WHERE experiment_hash = p_experiment_hash
     ORDER BY claim_generation DESC, created_at DESC, claim_key DESC
     LIMIT 1;
    IF NOT FOUND
       OR head.claim_state IS DISTINCT FROM 'claimed'
       OR head.claim_key IS DISTINCT FROM p_claim_key
       OR head.claim_generation IS DISTINCT FROM p_claim_generation
       OR head.claim_capability_commitment IS DISTINCT FROM expected_commitment
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_routing_experiment_claim_closures_v2 closed
           WHERE closed.experiment_hash = p_experiment_hash
             AND closed.claim_key = p_claim_key
             AND closed.claim_generation = p_claim_generation
             AND closed.claim_capability_commitment = expected_commitment
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_fence_stale' USING ERRCODE = '42501';
    END IF;
    SELECT greatest(
        head.lease_expires_at,
        coalesce(max(heartbeat.lease_expires_at), head.lease_expires_at)
    ) INTO effective_expiry
    FROM public.research_lab_routing_experiment_claim_heartbeats_v2 heartbeat
    WHERE heartbeat.experiment_hash = p_experiment_hash
      AND heartbeat.claim_key = p_claim_key
      AND heartbeat.claim_generation = p_claim_generation
      AND heartbeat.claim_capability_commitment = expected_commitment;
    IF effective_expiry <= pg_catalog.clock_timestamp() THEN
        RAISE EXCEPTION 'research_lab_routing_claim_fence_stale' USING ERRCODE = '42501';
    END IF;
END;
$assert_claim$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_submit_experiment_v2(
    p_experiment_hash TEXT,
    p_experiment_id TEXT,
    p_spec_doc JSONB,
    p_receipt_execution_mode TEXT,
    p_allow_live_credit_spend BOOLEAN,
    p_event_hash TEXT,
    p_event_doc JSONB,
    p_execution_envelope_hash TEXT DEFAULT NULL,
    p_execution_envelope_doc JSONB DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $submit$
DECLARE
    existing public.research_lab_routing_experiments_v2%ROWTYPE;
    result public.research_lab_routing_experiments_v2%ROWTYPE;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_receipt_execution_mode NOT IN ('fixture', 'replay', 'measured_lab')
       OR p_allow_live_credit_spend IS NULL
       OR p_spec_doc->>'experiment_id' IS DISTINCT FROM p_experiment_id
       OR p_spec_doc->>'receipt_execution_mode' IS DISTINCT FROM p_receipt_execution_mode
       OR (p_spec_doc->>'allow_live_credit_spend')::BOOLEAN IS DISTINCT FROM p_allow_live_credit_spend
       OR public.research_lab_routing_jsonb_hash_v2(p_spec_doc)
            IS DISTINCT FROM p_experiment_hash
       OR (p_execution_envelope_hash IS NULL) IS DISTINCT FROM (p_execution_envelope_doc IS NULL)
       OR (p_allow_live_credit_spend AND p_execution_envelope_hash IS NULL)
       OR (p_execution_envelope_hash IS NOT NULL
           AND p_execution_envelope_hash !~ '^sha256:[0-9a-f]{64}$')
       OR (p_execution_envelope_doc IS NOT NULL
           AND public.research_lab_routing_jsonb_hash_v2(p_execution_envelope_doc)
                IS DISTINCT FROM p_execution_envelope_hash)
       OR (p_execution_envelope_doc IS NOT NULL
           AND (
               pg_catalog.jsonb_typeof(p_execution_envelope_doc) <> 'object'
               OR p_execution_envelope_doc->>'schema_version'
                    IS DISTINCT FROM 'leadpoet.research_lab.routing_execution_envelope.v2'
               OR p_execution_envelope_doc->>'experiment_hash'
                    IS DISTINCT FROM p_experiment_hash
               OR p_execution_envelope_doc->>'binding_catalog_manifest_hash'
                    !~ '^sha256:[0-9a-f]{64}$'
               OR p_execution_envelope_doc->>'unit_dataset_manifest_hash'
                    !~ '^sha256:[0-9a-f]{64}$'
               OR p_execution_envelope_doc->>'unit_set_hash'
                    IS DISTINCT FROM p_spec_doc->'input'->>'unit_input_set_hash'
               OR p_execution_envelope_doc->>'gold_label_manifest_hash'
                    !~ '^sha256:[0-9a-f]{64}$'
               OR p_execution_envelope_doc->>'model_binding_observation_receipt_hash'
                    !~ '^sha256:[0-9a-f]{64}$'
               OR pg_catalog.jsonb_typeof(
                    p_execution_envelope_doc->'model_binding_observation'
                  ) <> 'object'
               OR p_execution_envelope_doc #>>
                    '{model_binding_observation,receipt,receipt_hash}'
                    IS DISTINCT FROM
                    p_execution_envelope_doc->>'model_binding_observation_receipt_hash'
               OR pg_catalog.jsonb_typeof(p_execution_envelope_doc->'bindings') <> 'array'
           ))
    THEN
        RAISE EXCEPTION 'research_lab_routing_submit_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_spec_doc, 'routing spec');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing event');
    IF p_execution_envelope_doc IS NOT NULL THEN
        PERFORM public.research_lab_routing_reject_secret_doc_v2(
            p_execution_envelope_doc, 'routing execution envelope'
        );
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    SELECT * INTO existing
      FROM public.research_lab_routing_experiments_v2
     WHERE experiment_hash = p_experiment_hash;
    IF FOUND THEN
        IF existing.experiment_id IS DISTINCT FROM p_experiment_id
           OR existing.spec_doc IS DISTINCT FROM p_spec_doc
           OR existing.receipt_execution_mode IS DISTINCT FROM p_receipt_execution_mode
           OR existing.allow_live_credit_spend IS DISTINCT FROM p_allow_live_credit_spend
           OR existing.execution_envelope_hash IS DISTINCT FROM p_execution_envelope_hash
           OR existing.execution_envelope_doc IS DISTINCT FROM p_execution_envelope_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_submit_conflict' USING ERRCODE = '23505';
        END IF;
        SELECT * INTO result FROM public.research_lab_routing_experiments_v2
         WHERE experiment_hash = p_experiment_hash;
    ELSE
        INSERT INTO public.research_lab_routing_experiments_v2 (
            experiment_hash, experiment_id, spec_doc, receipt_execution_mode,
            allow_live_credit_spend, execution_envelope_hash,
            execution_envelope_doc
        ) VALUES (
            p_experiment_hash, p_experiment_id, p_spec_doc,
            p_receipt_execution_mode, p_allow_live_credit_spend,
            p_execution_envelope_hash, p_execution_envelope_doc
        ) RETURNING * INTO result;
    END IF;
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, event_doc
    ) VALUES (p_event_hash, p_experiment_hash, 'submitted', p_event_doc)
    ON CONFLICT (event_hash) DO NOTHING;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_events_v2
        WHERE event_hash = p_event_hash
          AND (experiment_hash IS DISTINCT FROM p_experiment_hash
               OR event_type IS DISTINCT FROM 'submitted'
               OR event_doc IS DISTINCT FROM p_event_doc)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_submit_event_conflict' USING ERRCODE = '23505';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'experiment_hash', result.experiment_hash,
        'experiment_id', result.experiment_id,
        'created_at', result.created_at,
        'idempotent', existing.experiment_hash IS NOT NULL
    );
END;
$submit$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_request_execution_v2(
    p_request_hash TEXT,
    p_experiment_hash TEXT,
    p_request_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $request_execution$
DECLARE
    existing public.research_lab_routing_execution_requests_v2%ROWTYPE;
BEGIN
    IF p_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_request_doc) <> 'object'
       OR p_request_doc->>'schema_version'
            IS DISTINCT FROM 'leadpoet.research_lab.routing_execution_request.v2'
       OR p_request_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
    THEN
        RAISE EXCEPTION 'research_lab_routing_request_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_request_doc, 'routing execution request'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    IF NOT EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiments_v2 experiment
        WHERE experiment.experiment_hash = p_experiment_hash
          AND (
              NOT experiment.allow_live_credit_spend
              OR experiment.execution_envelope_hash IS NOT NULL
          )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_request_experiment_unavailable' USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_claim_closures_v2 closure
        WHERE closure.experiment_hash = p_experiment_hash
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_routing_experiment_claims_v2 recovered_claim
        WHERE recovered_claim.experiment_hash = p_experiment_hash
          AND recovered_claim.claim_state = 'recovered'
    ) OR EXISTS (
        SELECT 1 FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
        WHERE evaluation.experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_request_experiment_terminal' USING ERRCODE = '23505';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_routing_execution_requests_v2
     WHERE experiment_hash = p_experiment_hash;
    IF FOUND THEN
        IF existing.request_hash IS DISTINCT FROM p_request_hash
           OR existing.request_doc IS DISTINCT FROM p_request_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_request_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'request_hash', existing.request_hash,
            'experiment_hash', existing.experiment_hash,
            'idempotent', TRUE
        );
    END IF;
    INSERT INTO public.research_lab_routing_execution_requests_v2 (
        request_hash, experiment_hash, request_doc
    ) VALUES (p_request_hash, p_experiment_hash, p_request_doc);
    RETURN pg_catalog.jsonb_build_object(
        'request_hash', p_request_hash,
        'experiment_hash', p_experiment_hash,
        'idempotent', FALSE
    );
END;
$request_execution$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_claim_experiment_v2(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_nonce TEXT,
    p_worker_ref TEXT,
    p_lease_seconds INTEGER,
    p_claim_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $claim$
DECLARE
    existing public.research_lab_routing_experiment_claims_v2%ROWTYPE;
    head public.research_lab_routing_experiment_claims_v2%ROWTYPE;
    expiry TIMESTAMPTZ;
    next_generation BIGINT;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_lease_seconds < 1 OR p_lease_seconds > 3600
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_claim_doc, 'routing claim');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing claim event');
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    IF NOT EXISTS (SELECT 1 FROM public.research_lab_routing_experiments_v2 WHERE experiment_hash = p_experiment_hash) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_experiment_missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing FROM public.research_lab_routing_experiment_claims_v2
     WHERE claim_key = p_claim_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.worker_ref IS DISTINCT FROM p_worker_ref
           OR existing.claim_capability_commitment IS DISTINCT FROM
              public.research_lab_routing_claim_capability_commitment_v2(p_claim_nonce)
           OR existing.claim_state IS DISTINCT FROM 'claimed'
           OR existing.claim_doc IS DISTINCT FROM p_claim_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_conflict' USING ERRCODE = '23505';
        END IF;
        IF EXISTS (
            SELECT 1
            FROM public.research_lab_routing_experiment_claim_closures_v2 closure
            WHERE closure.experiment_hash = p_experiment_hash
              AND closure.claim_key = existing.claim_key
              AND closure.claim_generation = existing.claim_generation
        ) OR EXISTS (
            SELECT 1
            FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
            WHERE evaluation.experiment_hash = p_experiment_hash
        ) THEN
            -- An exact retry may learn that its prior claim reached a terminal
            -- state. It receives no active lease and cannot mutate anything.
            RETURN pg_catalog.jsonb_build_object(
                'claimed', FALSE, 'idempotent', TRUE, 'terminal', TRUE,
                'recoverable', FALSE,
                'claim_key', existing.claim_key,
                'claim_generation', existing.claim_generation,
                'lease_expires_at', existing.lease_expires_at
            );
        END IF;
        IF existing.lease_expires_at <= pg_catalog.clock_timestamp() THEN
            -- A caller must record a fenced recovery before it can begin a
            -- fresh generation.  Returning this state rather than silently
            -- creating a new claim keeps crash recovery auditable.
            RETURN pg_catalog.jsonb_build_object(
                'claimed', FALSE, 'idempotent', FALSE, 'recoverable', TRUE,
                'claim_key', existing.claim_key,
                'claim_generation', existing.claim_generation,
                'lease_expires_at', existing.lease_expires_at
            );
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'claimed', TRUE, 'idempotent', TRUE,
            'claim_key', existing.claim_key,
            'claim_generation', existing.claim_generation,
            'lease_expires_at', existing.lease_expires_at
        );
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_routing_experiment_claim_closures_v2 closure
        WHERE closure.experiment_hash = p_experiment_hash
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
        WHERE evaluation.experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_experiment_terminal' USING ERRCODE = '23505';
    END IF;
    SELECT * INTO head FROM public.research_lab_routing_experiment_claims_v2
     WHERE experiment_hash = p_experiment_hash
     ORDER BY claim_generation DESC, created_at DESC, claim_key DESC LIMIT 1;
    IF FOUND AND head.claim_state = 'recovered' THEN
        -- Recovery is terminal. Unknown billing requires a new immutable
        -- experiment; this authority never retries under the old budget.
        RAISE EXCEPTION 'research_lab_routing_claim_experiment_recovered' USING ERRCODE = '23505';
    END IF;
    IF FOUND AND head.claim_state = 'claimed' THEN
        IF head.lease_expires_at > pg_catalog.clock_timestamp() THEN
            RETURN pg_catalog.jsonb_build_object(
                'claimed', FALSE, 'idempotent', FALSE,
                'claim_key', head.claim_key, 'claim_generation', head.claim_generation,
                'lease_expires_at', head.lease_expires_at
            );
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'claimed', FALSE, 'idempotent', FALSE, 'recoverable', TRUE,
            'claim_key', head.claim_key, 'claim_generation', head.claim_generation,
            'lease_expires_at', head.lease_expires_at
        );
    END IF;
    SELECT coalesce(max(claim_generation), 0) + 1 INTO next_generation
      FROM public.research_lab_routing_experiment_claims_v2
     WHERE experiment_hash = p_experiment_hash;
    expiry := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_seconds);
    INSERT INTO public.research_lab_routing_experiment_claims_v2 (
        claim_key, experiment_hash, claim_generation, claim_capability_commitment,
        worker_ref, claim_state, lease_expires_at, claim_doc
    ) VALUES (
        p_claim_key, p_experiment_hash, next_generation,
        public.research_lab_routing_claim_capability_commitment_v2(p_claim_nonce),
        p_worker_ref, 'claimed', expiry, p_claim_doc
    );
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, event_doc
    ) VALUES (p_event_hash, p_experiment_hash, 'claimed', p_event_doc)
    ON CONFLICT (event_hash) DO NOTHING;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_events_v2
        WHERE event_hash = p_event_hash
          AND (experiment_hash IS DISTINCT FROM p_experiment_hash
               OR event_type IS DISTINCT FROM 'claimed'
               OR event_doc IS DISTINCT FROM p_event_doc)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_event_conflict' USING ERRCODE = '23505';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'claimed', TRUE, 'idempotent', FALSE,
        'claim_key', p_claim_key,
        'claim_generation', next_generation,
        'lease_expires_at', expiry
    );
END;
$claim$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_renew_claim_v2(
    p_heartbeat_key TEXT,
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_lease_seconds INTEGER,
    p_heartbeat_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $renew_claim$
DECLARE
    commitment TEXT;
    expiry TIMESTAMPTZ;
    existing public.research_lab_routing_experiment_claim_heartbeats_v2%ROWTYPE;
BEGIN
    IF p_heartbeat_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation < 1
       OR p_lease_seconds < 1 OR p_lease_seconds > 3600
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_renew_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    commitment := public.research_lab_routing_claim_capability_commitment_v2(p_claim_nonce);
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_heartbeat_doc, 'routing claim heartbeat');
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    expiry := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_seconds);
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claim_heartbeats_v2
     WHERE heartbeat_key = p_heartbeat_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.claim_capability_commitment IS DISTINCT FROM commitment
           OR existing.heartbeat_doc IS DISTINCT FROM p_heartbeat_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_renew_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'renewed', TRUE, 'idempotent', TRUE,
            'heartbeat_key', p_heartbeat_key,
            'lease_expires_at', existing.lease_expires_at
        );
    END IF;
    INSERT INTO public.research_lab_routing_experiment_claim_heartbeats_v2 (
        heartbeat_key, experiment_hash, claim_key, claim_generation,
        claim_capability_commitment, lease_expires_at, heartbeat_doc
    ) VALUES (
        p_heartbeat_key, p_experiment_hash, p_claim_key, p_claim_generation,
        commitment, expiry, p_heartbeat_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'renewed', TRUE, 'idempotent', FALSE,
        'heartbeat_key', p_heartbeat_key, 'lease_expires_at', expiry
    );
END;
$renew_claim$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_close_claim_v2(
    p_close_key TEXT,
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_close_reason TEXT,
    p_close_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $close_claim$
DECLARE
    commitment TEXT;
    existing public.research_lab_routing_experiment_claim_closures_v2%ROWTYPE;
BEGIN
    IF p_close_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation < 1
       OR p_close_reason NOT IN ('completed', 'failed', 'cancelled')
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_close_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    commitment := public.research_lab_routing_claim_capability_commitment_v2(p_claim_nonce);
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_close_doc, 'routing claim close');
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claim_closures_v2
     WHERE close_key = p_close_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.claim_capability_commitment IS DISTINCT FROM commitment
           OR existing.close_reason IS DISTINCT FROM p_close_reason
           OR existing.close_doc IS DISTINCT FROM p_close_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_close_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('closed', TRUE, 'idempotent', TRUE, 'close_key', p_close_key);
    END IF;
    INSERT INTO public.research_lab_routing_experiment_claim_closures_v2 (
        close_key, experiment_hash, claim_key, claim_generation,
        claim_capability_commitment, close_reason, close_doc
    ) VALUES (
        p_close_key, p_experiment_hash, p_claim_key, p_claim_generation,
        commitment, p_close_reason, p_close_doc
    );
    RETURN pg_catalog.jsonb_build_object('closed', TRUE, 'idempotent', FALSE, 'close_key', p_close_key);
END;
$close_claim$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_event_v2(
    p_event_hash TEXT,
    p_experiment_hash TEXT,
    p_event_type TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $append_event$
DECLARE
    existing public.research_lab_routing_experiment_events_v2%ROWTYPE;
BEGIN
    IF p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_type NOT IN (
           'run_started', 'run_completed', 'run_failed', 'promotion_requested'
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_event_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing event');
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    IF NOT EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiments_v2
        WHERE experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_event_experiment_missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing FROM public.research_lab_routing_experiment_events_v2
     WHERE event_hash = p_event_hash;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.event_type IS DISTINCT FROM p_event_type
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_event_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('event_hash', p_event_hash, 'idempotent', TRUE);
    END IF;
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, event_doc
    ) VALUES (p_event_hash, p_experiment_hash, p_event_type, p_event_doc);
    RETURN pg_catalog.jsonb_build_object('event_hash', p_event_hash, 'idempotent', FALSE);
END;
$append_event$;

-- The historical append_event_v2 signature remains installed for migration
-- compatibility but is not executable by service_role.  Worker lifecycle
-- writes use this separately named, claim-fenced authority instead.
CREATE OR REPLACE FUNCTION public.research_lab_routing_append_fenced_event_v2(
    p_event_hash TEXT,
    p_experiment_hash TEXT,
    p_event_type TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $append_fenced_event$
DECLARE
    existing public.research_lab_routing_experiment_events_v2%ROWTYPE;
BEGIN
    IF p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_type NOT IN (
           'run_started', 'run_completed', 'run_failed', 'promotion_requested',
           'provider_dispatch_started'
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_event_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing event');
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    IF p_event_type = 'provider_dispatch_started' THEN
        IF p_event_doc->>'reservation_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'binding_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'request_fingerprint' !~ '^sha256:[0-9a-f]{64}$'
           OR NOT EXISTS (
               SELECT 1
               FROM public.research_lab_routing_budget_events_v2 reserve_event
               WHERE reserve_event.experiment_hash = p_experiment_hash
                 AND reserve_event.reservation_id = p_event_doc->>'reservation_id'
                 AND reserve_event.binding_id = p_event_doc->>'binding_id'
                 AND reserve_event.claim_key = p_claim_key
                 AND reserve_event.claim_generation = p_claim_generation
                 AND reserve_event.event_type = 'reserve'
           )
        THEN
            RAISE EXCEPTION 'research_lab_routing_dispatch_event_reservation_mismatch'
                USING ERRCODE = '23503';
        END IF;
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    IF NOT EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiments_v2
        WHERE experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_event_experiment_missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing FROM public.research_lab_routing_experiment_events_v2
     WHERE event_hash = p_event_hash;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.event_type IS DISTINCT FROM p_event_type
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_event_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('event_hash', p_event_hash, 'idempotent', TRUE);
    END IF;
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, claim_key, claim_generation, event_doc
    ) VALUES (
        p_event_hash, p_experiment_hash, p_event_type,
        p_claim_key, p_claim_generation, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object('event_hash', p_event_hash, 'idempotent', FALSE);
END;
$append_fenced_event$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_recover_claim_v2(
    p_experiment_hash TEXT,
    p_recovery_key TEXT,
    p_worker_ref TEXT,
    p_recovery_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $recover_claim$
DECLARE
    head public.research_lab_routing_experiment_claims_v2%ROWTYPE;
    existing public.research_lab_routing_experiment_claims_v2%ROWTYPE;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_recovery_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_recover_claim_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_recovery_doc, 'routing recovery');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing recovery event');
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_routing_experiment_claim_closures_v2 closure
        WHERE closure.experiment_hash = p_experiment_hash
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
        WHERE evaluation.experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_recover_claim_experiment_terminal' USING ERRCODE = '23505';
    END IF;
    -- Recovery has its own immutable identity.  A timeout after the first
    -- transaction committed must return the original recovery, even when a
    -- later worker has already claimed the next generation.
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claims_v2
     WHERE claim_key = p_recovery_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.worker_ref IS DISTINCT FROM p_worker_ref
           OR existing.claim_state IS DISTINCT FROM 'recovered'
           OR existing.claim_capability_commitment IS DISTINCT FROM 'sha256:0000000000000000000000000000000000000000000000000000000000000000'
           OR existing.claim_doc IS DISTINCT FROM p_recovery_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_recover_claim_conflict' USING ERRCODE = '23505';
        END IF;
        IF EXISTS (
            SELECT 1 FROM public.research_lab_routing_experiment_events_v2 event
             WHERE event.event_hash = p_event_hash
               AND (
                   event.experiment_hash IS DISTINCT FROM p_experiment_hash
                   OR event.event_type IS DISTINCT FROM 'claim_recovered'
                   OR event.event_doc IS DISTINCT FROM p_event_doc
               )
        ) THEN
            RAISE EXCEPTION 'research_lab_routing_recover_claim_event_conflict' USING ERRCODE = '23505';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_experiment_events_v2 event
             WHERE event.event_hash = p_event_hash
               AND event.experiment_hash = p_experiment_hash
               AND event.event_type = 'claim_recovered'
               AND event.event_doc = p_event_doc
        ) THEN
            RAISE EXCEPTION 'research_lab_routing_recover_claim_event_missing' USING ERRCODE = '23503';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'recovered', TRUE,
            'idempotent', TRUE,
            'recovery_key', p_recovery_key,
            'claim_generation', existing.claim_generation
        );
    END IF;
    SELECT * INTO head FROM public.research_lab_routing_experiment_claims_v2
     WHERE experiment_hash = p_experiment_hash
     ORDER BY claim_generation DESC, created_at DESC, claim_key DESC LIMIT 1;
    IF NOT FOUND OR head.claim_state IS DISTINCT FROM 'claimed' OR head.lease_expires_at >= pg_catalog.clock_timestamp() THEN
        RAISE EXCEPTION 'research_lab_routing_recover_claim_not_expired' USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.research_lab_routing_experiment_claims_v2 (
        claim_key, experiment_hash, claim_generation, claim_capability_commitment,
        worker_ref, claim_state, claim_doc
    ) VALUES (
        p_recovery_key, p_experiment_hash, head.claim_generation + 1,
        'sha256:0000000000000000000000000000000000000000000000000000000000000000',
        p_worker_ref, 'recovered', p_recovery_doc
    );
    -- A recovered experiment is terminal. Every open reservation keeps its
    -- full ceiling as an append-only uncertain head. No later generation can
    -- release or retry spend whose provider billing is not authoritative.
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type, credit_microunits, event_doc
    )
    SELECT
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.research_lab.routing_budget_recovery.v2',
                'event_type', 'uncertain',
                'reservation_id', head_event.reservation_id,
                'recovery_key', p_recovery_key,
                'reserve_event_key', head_event.event_key
            )
        ),
        head_event.reservation_id,
        p_experiment_hash,
        head_event.binding_id,
        p_recovery_key,
        head.claim_generation + 1,
        'uncertain',
        head_event.credit_microunits,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_budget_event.v2',
            'billing_state', 'uncertain',
            'reason_code', 'claim_recovered_unknown_billing',
            'reservation_id', head_event.reservation_id,
            'binding_id', head_event.binding_id,
            'recovery_key', p_recovery_key,
            'reserve_event_key', head_event.event_key
        )
    FROM (
        SELECT DISTINCT ON (budget_event.reservation_id) budget_event.*
        FROM public.research_lab_routing_budget_events_v2 budget_event
        WHERE budget_event.experiment_hash = p_experiment_hash
        ORDER BY budget_event.reservation_id,
            budget_event.created_at DESC, budget_event.event_key DESC
    ) AS head_event
    -- Claim recovery is terminal for the experiment.  A reservation can
    -- still have a later lease than the claim, but its provider call is
    -- nevertheless no longer fenced by a live worker.  Preserve its full
    -- ceiling as uncertain instead of allowing a later worker to spend it.
    WHERE head_event.event_type = 'reserve';
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, event_doc
    ) VALUES (p_event_hash, p_experiment_hash, 'claim_recovered', p_event_doc)
    ON CONFLICT (event_hash) DO NOTHING;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_events_v2
        WHERE event_hash = p_event_hash
          AND (experiment_hash IS DISTINCT FROM p_experiment_hash
               OR event_type IS DISTINCT FROM 'claim_recovered'
               OR event_doc IS DISTINCT FROM p_event_doc)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_recover_claim_event_conflict' USING ERRCODE = '23505';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'recovered', TRUE,
        'idempotent', FALSE,
        'recovery_key', p_recovery_key,
        'claim_generation', head.claim_generation + 1
    );
END;
$recover_claim$;

-- Check the signed receipt links before a provider attempt is persisted.  The
-- helper is SECURITY DEFINER and intentionally has no public grant; only the
-- append RPC below calls it after claim and reservation fencing.
CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v2(
    p_experiment_hash TEXT,
    p_binding_id TEXT,
    p_tool_id TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_action_id TEXT,
    p_authorization_hash TEXT,
    p_authorization_proof_hash TEXT,
    p_terminal_receipt_hash TEXT,
    p_protected_release_receipt_hash TEXT,
    p_admission_bundle_hash TEXT,
    p_terminal_provider_record_hash TEXT,
    p_terminal_billing_projection_hash TEXT,
    p_outcome TEXT,
    p_credit_microunits BIGINT,
    p_latency_ms BIGINT,
    p_billing_state TEXT,
    p_authoritative_billed_credit_microunits BIGINT,
    p_attempt_doc JSONB
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $receipt_chain$
DECLARE
    authorization_doc JSONB := p_attempt_doc->'call_grant_receipt';
    protected_doc JSONB := p_attempt_doc->'protected_release_receipt';
    terminal_doc JSONB := p_attempt_doc->'terminal_execution_receipt';
    terminal_result JSONB := p_attempt_doc->'terminal_result';
    admission JSONB := p_attempt_doc->'admission_bundle';
    call_grant_doc JSONB := p_attempt_doc->'call_grant';
    grant_result JSONB := p_attempt_doc->'call_grant_result';
    experiment_envelope JSONB;
    experiment_envelope_hash TEXT;
    model_observation_hash TEXT;
    terminal_job_id TEXT;
    billing_projection JSONB;
BEGIN
    IF pg_catalog.jsonb_typeof(authorization_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(protected_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(terminal_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(terminal_result) <> 'object'
       OR pg_catalog.jsonb_typeof(admission) <> 'object'
       OR pg_catalog.jsonb_typeof(call_grant_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(grant_result) <> 'object'
       OR p_attempt_doc->>'schema_version' IS DISTINCT FROM 'leadpoet.research_lab.routing_provider_attempt.v3'
       OR p_attempt_doc->>'authorization_request_hash' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_receipt_graph_shape_invalid'
            USING ERRCODE = '23503';
    END IF;
    SELECT experiment.execution_envelope_doc INTO experiment_envelope
      FROM public.research_lab_routing_experiments_v2 experiment
     WHERE experiment.experiment_hash = p_experiment_hash;
    model_observation_hash := experiment_envelope->>'model_binding_observation_receipt_hash';
    IF model_observation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR admission->>'model_binding_observation_receipt_hash' IS DISTINCT FROM model_observation_hash
       OR call_grant_doc->>'admission_job_id' IS DISTINCT FROM admission->>'job_id'
       OR public.research_lab_routing_jsonb_hash_v2(admission) IS DISTINCT FROM p_admission_bundle_hash
       OR admission->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR call_grant_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR call_grant_doc->>'admission_bundle_hash' IS DISTINCT FROM p_admission_bundle_hash
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_admission_binding_mismatch'
            USING ERRCODE = '22023';
    END IF;
    -- The protected authorization execution is a distinct job from durable
    -- admission. Its receipt must bind the exact request and result roots.
    IF grant_result->>'authorization_job_id' IS NULL
       OR grant_result->>'authorization_job_id' = admission->>'job_id'
       OR grant_result->>'authorization_job_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR grant_result->>'authorization_hash' IS DISTINCT FROM p_authorization_hash
       OR p_attempt_doc->>'authorization_request_hash' IS DISTINCT FROM authorization_doc->>'input_root'
       OR authorization_doc->>'receipt_hash' IS DISTINCT FROM p_authorization_proof_hash
       OR authorization_doc->>'job_id' IS DISTINCT FROM grant_result->>'authorization_job_id'
       OR authorization_doc->>'input_root' IS DISTINCT FROM p_attempt_doc->>'authorization_request_hash'
       OR authorization_doc->>'output_root' IS DISTINCT FROM grant_result->>'output_root'
       OR authorization_doc->'parent_receipt_hashes' IS DISTINCT FROM
            pg_catalog.jsonb_build_array(p_protected_release_receipt_hash, model_observation_hash)
       OR pg_catalog.jsonb_array_length(authorization_doc->'parent_receipt_hashes') <> 2
       OR (SELECT count(DISTINCT parent_hash.value)
             FROM pg_catalog.jsonb_array_elements_text(authorization_doc->'parent_receipt_hashes') AS parent_hash(value)) <> 2
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_authorization_binding_mismatch'
            USING ERRCODE = '22023';
    END IF;
    IF pg_catalog.jsonb_typeof(authorization_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(protected_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(terminal_doc) <> 'object'
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_receipt_graph_shape_invalid'
            USING ERRCODE = '23503';
    END IF;
    billing_projection := terminal_result->'projection';
    IF pg_catalog.jsonb_typeof(billing_projection) <> 'object'
       OR terminal_result->>'authorization_hash' IS DISTINCT FROM p_authorization_hash
       OR terminal_result->>'authorization_proof_hash' IS DISTINCT FROM p_authorization_proof_hash
       OR terminal_result->>'provider_record_hash' IS DISTINCT FROM p_terminal_provider_record_hash
       OR p_terminal_billing_projection_hash IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(billing_projection)
       OR terminal_result->'binding'->>'binding_id' IS DISTINCT FROM p_binding_id
       OR terminal_result->'binding'->>'tool_id' IS DISTINCT FROM p_tool_id
       OR terminal_result->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR terminal_result->>'request_fingerprint' IS DISTINCT FROM p_attempt_doc->>'request_fingerprint'
       OR billing_projection->>'outcome' IS NULL
       OR billing_projection->>'outcome' IS DISTINCT FROM p_outcome
       OR billing_projection->>'credit_microunits' IS NULL
       OR (billing_projection->>'credit_microunits')::BIGINT IS DISTINCT FROM p_credit_microunits
       OR billing_projection->>'latency_ms' IS NULL
       OR (billing_projection->>'latency_ms')::BIGINT IS DISTINCT FROM p_latency_ms
       OR billing_projection->>'billing_state' IS NULL
       OR billing_projection->>'billing_state' IS DISTINCT FROM 'known'
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_authoritative_billing_mismatch'
            USING ERRCODE = '22023';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        JOIN public.research_lab_attested_boot_identities_v2 signer
          ON signer.boot_identity_hash = receipt.boot_identity_hash
        WHERE receipt.receipt_hash = p_authorization_proof_hash
          AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
          AND receipt.role = 'gateway_scoring'
          AND receipt.purpose = 'research_lab.routing_provider_evidence.v2'
          AND receipt.receipt_status = 'succeeded'
          AND receipt.job_id = grant_result->>'authorization_job_id'
          AND receipt.input_root = p_attempt_doc->>'authorization_request_hash'
          AND receipt.output_root = grant_result->>'output_root'
          AND receipt.receipt_doc = p_attempt_doc->'call_grant_receipt'
          AND receipt.receipt_doc->>'receipt_hash' = receipt.receipt_hash
          AND receipt.receipt_doc->>'role' = receipt.role
          AND receipt.receipt_doc->>'purpose' = receipt.purpose
          AND receipt.receipt_doc->>'job_id' = receipt.job_id
          AND receipt.receipt_doc->>'input_root' = receipt.input_root
          AND receipt.receipt_doc->>'output_root' = receipt.output_root
          AND receipt.receipt_doc->>'status' = 'succeeded'
          AND receipt.receipt_doc->'parent_receipt_hashes' = pg_catalog.jsonb_build_array(
              p_protected_release_receipt_hash, model_observation_hash
          )
          AND receipt.enclave_pubkey = signer.signing_pubkey
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_authorization_receipt_missing' USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        JOIN public.research_lab_attested_boot_identities_v2 signer
          ON signer.boot_identity_hash = receipt.boot_identity_hash
        WHERE receipt.receipt_hash = p_protected_release_receipt_hash
          AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
          AND receipt.role = 'gateway_scoring'
          AND receipt.purpose = 'research_lab.routing_provider_evidence.v2'
          AND receipt.receipt_status = 'succeeded'
          AND receipt.job_id = p_attempt_doc->'admission_bundle'->>'job_id'
          AND receipt.receipt_doc = p_attempt_doc->'protected_release_receipt'
          AND receipt.receipt_doc->>'receipt_hash' = receipt.receipt_hash
          AND receipt.receipt_doc->>'role' = receipt.role
          AND receipt.receipt_doc->>'purpose' = receipt.purpose
          AND receipt.receipt_doc->>'job_id' = receipt.job_id
          AND receipt.receipt_doc->>'input_root' = receipt.input_root
          AND receipt.receipt_doc->>'output_root' = receipt.output_root
          AND receipt.receipt_doc->>'status' = 'succeeded'
          AND receipt.receipt_doc->'parent_receipt_hashes' = admission->'parent_receipt_hashes'
          AND receipt.enclave_pubkey = signer.signing_pubkey
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_protected_release_receipt_missing' USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        JOIN public.research_lab_attested_boot_identities_v2 signer
          ON signer.boot_identity_hash = receipt.boot_identity_hash
        WHERE receipt.receipt_hash = p_terminal_receipt_hash
          AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
          AND receipt.role = 'gateway_scoring'
          AND receipt.purpose = 'research_lab.routing_provider_evidence.v2'
          AND receipt.receipt_status = 'succeeded'
          AND receipt.job_id = terminal_doc->>'job_id'
          AND receipt.job_id <> admission->>'job_id'
          AND receipt.job_id <> grant_result->>'authorization_job_id'
          AND receipt.input_root = p_attempt_doc->>'terminal_request_hash'
          AND receipt.output_root = public.research_lab_routing_jsonb_hash_v2(terminal_result)
          AND receipt.receipt_doc = terminal_doc
          AND receipt.receipt_doc->>'receipt_hash' = receipt.receipt_hash
          AND receipt.receipt_doc->>'role' = receipt.role
          AND receipt.receipt_doc->>'purpose' = receipt.purpose
          AND receipt.receipt_doc->>'job_id' = receipt.job_id
          AND receipt.receipt_doc->>'input_root' = receipt.input_root
          AND receipt.receipt_doc->>'output_root' = receipt.output_root
          AND receipt.receipt_doc->>'status' = 'succeeded'
          AND receipt.receipt_doc->'parent_receipt_hashes' = pg_catalog.jsonb_build_array(
              p_authorization_proof_hash
          )
          AND receipt.enclave_pubkey = signer.signing_pubkey
          AND receipt.boot_identity_hash = (
              SELECT protected.boot_identity_hash
              FROM public.research_lab_attested_execution_receipts_v2 protected
              WHERE protected.receipt_hash = p_protected_release_receipt_hash
          )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_terminal_receipt_missing' USING ERRCODE = '23503';
    END IF;
    -- All three standard receipts must come from the same installed scoring
    -- boot identity.  A valid signature from another scorer is not valid for
    -- this admission, even when all JSON commitments happen to match.
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 authorization_receipt
        JOIN public.research_lab_attested_execution_receipts_v2 protected_receipt
          ON protected_receipt.receipt_hash = p_protected_release_receipt_hash
        JOIN public.research_lab_attested_execution_receipts_v2 terminal_receipt
          ON terminal_receipt.receipt_hash = p_terminal_receipt_hash
        WHERE authorization_receipt.receipt_hash = p_authorization_proof_hash
          AND authorization_receipt.boot_identity_hash = protected_receipt.boot_identity_hash
          AND authorization_receipt.boot_identity_hash = terminal_receipt.boot_identity_hash
          AND authorization_receipt.enclave_pubkey = protected_receipt.enclave_pubkey
          AND authorization_receipt.enclave_pubkey = terminal_receipt.enclave_pubkey
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_receipt_signer_mismatch'
            USING ERRCODE = '23503';
    END IF;
END;
$receipt_chain$;
-- A database that already ran the earlier 14-argument helper may still have
-- that overload.  Move it out of the public name as well; otherwise an RPC
-- resolver could select the legacy implementation on replay.
DO $retire_legacy_receipt_overload$
BEGIN
    IF pg_catalog.to_regprocedure(
        'public.research_lab_routing_assert_provider_receipt_chain_v2(text,text,text,text,text,text,text,text,text,text,text,text,text,jsonb)'
    ) IS NOT NULL THEN
        EXECUTE 'ALTER FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v2(text,text,text,text,text,text,text,text,text,text,text,text,text,jsonb) RENAME TO research_lab_routing_assert_provider_receipt_chain_legacy_v1';
        EXECUTE 'REVOKE ALL ON FUNCTION public.research_lab_routing_assert_provider_receipt_chain_legacy_v1(text,text,text,text,text,text,text,text,text,text,text,text,text,jsonb) FROM PUBLIC, anon, authenticated, service_role';
    END IF;
END;
$retire_legacy_receipt_overload$;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v2(
    TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT,
    TEXT, BIGINT, BIGINT, TEXT, BIGINT, JSONB
) FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_provider_attempt_v2(
    p_attempt_key TEXT,
    p_experiment_hash TEXT,
    p_provider_receipt_ref TEXT,
    p_binding_id TEXT,
    p_tool_id TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_reservation_id TEXT,
    p_action_id TEXT,
    p_binding_catalog_manifest_hash TEXT,
    p_authorization_hash TEXT,
    p_authorization_proof_hash TEXT,
    p_request_body_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_request_fingerprint TEXT,
    p_outcome TEXT,
    p_credit_microunits BIGINT,
    p_latency_ms BIGINT,
    p_execution_mode TEXT,
    p_billing_state TEXT,
    p_authoritative_billed_credit_microunits BIGINT,
    p_terminal_receipt_hash TEXT,
    p_protected_release_receipt_hash TEXT,
    p_admission_bundle_hash TEXT,
    p_terminal_provider_record_hash TEXT,
    p_terminal_billing_projection_hash TEXT,
    p_attempt_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $attempt$
DECLARE
    existing public.research_lab_routing_provider_attempts_v2%ROWTYPE;
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    billing_projection JSONB;
BEGIN
    IF p_attempt_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_provider_receipt_ref !~ '^provider_receipt:[0-9a-f]{16}$'
       OR p_binding_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_tool_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_unit_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_action_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_binding_catalog_manifest_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_proof_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_request_body_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_receipt_hash IS NULL
       OR p_terminal_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_protected_release_receipt_hash IS NULL
       OR p_protected_release_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_admission_bundle_hash IS NULL
       OR p_admission_bundle_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_provider_record_hash IS NULL
       OR p_terminal_provider_record_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_billing_projection_hash IS NULL
       OR p_terminal_billing_projection_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_request_fingerprint !~ '^sha256:[0-9a-f]{64}$'
       OR p_outcome NOT IN ('verified', 'rejected', 'source_miss', 'adapter_failure')
       OR p_credit_microunits < 0 OR p_credit_microunits > 10000000
       OR p_latency_ms < 0 OR p_latency_ms > 900000
       OR p_execution_mode NOT IN ('fixture', 'replay', 'measured_lab')
       OR p_billing_state NOT IN ('known', 'uncertain')
       OR (p_billing_state = 'known'
           AND (p_authoritative_billed_credit_microunits IS NULL
                OR p_authoritative_billed_credit_microunits < 0
                OR p_authoritative_billed_credit_microunits > 10000000))
       OR (p_billing_state = 'uncertain'
           AND p_authoritative_billed_credit_microunits IS NOT NULL)
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    IF p_outcome = 'adapter_failure' AND p_credit_microunits <> 0 THEN
        RAISE EXCEPTION 'research_lab_routing_adapter_failure_must_be_zero_cost' USING ERRCODE = '22023';
    END IF;
    IF p_outcome <> 'adapter_failure'
       AND (p_billing_state IS DISTINCT FROM 'known'
            OR p_authoritative_billed_credit_microunits IS DISTINCT FROM p_credit_microunits)
    THEN
        RAISE EXCEPTION 'research_lab_routing_nonfailure_billing_must_match_receipt' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_attempt_doc, 'routing provider attempt');
    -- A replay may be checked before the live claim/reservation lifecycle,
    -- but only after the same proof-chain validator has revalidated the
    -- immutable stored row.  This keeps restart idempotency while preserving
    -- fail-closed authority checks for legacy or corrupted rows.
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_attempt_key, 0));
    SELECT * INTO existing
      FROM public.research_lab_routing_provider_attempts_v2
     WHERE attempt_key = p_attempt_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.provider_receipt_ref IS DISTINCT FROM p_provider_receipt_ref
           OR existing.binding_id IS DISTINCT FROM p_binding_id
           OR existing.tool_id IS DISTINCT FROM p_tool_id
           OR existing.variant_id IS DISTINCT FROM p_variant_id
           OR existing.unit_ref IS DISTINCT FROM p_unit_ref
           OR existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.action_id IS DISTINCT FROM p_action_id
           OR existing.binding_catalog_manifest_hash IS DISTINCT FROM p_binding_catalog_manifest_hash
           OR existing.authorization_hash IS DISTINCT FROM p_authorization_hash
           OR existing.authorization_proof_hash IS DISTINCT FROM p_authorization_proof_hash
           OR existing.request_body_hash IS DISTINCT FROM p_request_body_hash
           OR existing.terminal_receipt_hash IS DISTINCT FROM p_terminal_receipt_hash
           OR existing.protected_release_receipt_hash IS DISTINCT FROM p_protected_release_receipt_hash
           OR existing.admission_bundle_hash IS DISTINCT FROM p_admission_bundle_hash
           OR existing.terminal_provider_record_hash IS DISTINCT FROM p_terminal_provider_record_hash
           OR existing.terminal_billing_projection_hash IS DISTINCT FROM p_terminal_billing_projection_hash
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.request_fingerprint IS DISTINCT FROM p_request_fingerprint
           OR existing.outcome IS DISTINCT FROM p_outcome
           OR existing.credit_microunits IS DISTINCT FROM p_credit_microunits
           OR existing.billing_state IS DISTINCT FROM p_billing_state
           OR existing.authoritative_billed_credit_microunits IS DISTINCT FROM p_authoritative_billed_credit_microunits
           OR existing.latency_ms IS DISTINCT FROM p_latency_ms
           OR existing.execution_mode IS DISTINCT FROM p_execution_mode
           OR existing.attempt_doc IS DISTINCT FROM p_attempt_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_attempt_conflict' USING ERRCODE = '23505';
        END IF;
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
            existing.experiment_hash, existing.binding_id, existing.tool_id,
            existing.variant_id, existing.unit_ref, existing.action_id,
            existing.authorization_hash, existing.authorization_proof_hash,
            existing.terminal_receipt_hash,
            existing.protected_release_receipt_hash,
            existing.admission_bundle_hash,
            existing.terminal_provider_record_hash,
            existing.terminal_billing_projection_hash,
            existing.outcome, existing.credit_microunits,
            existing.latency_ms, existing.billing_state,
            existing.authoritative_billed_credit_microunits,
            existing.attempt_doc
        );
        RETURN pg_catalog.jsonb_build_object('attempt_key', existing.attempt_key, 'idempotent', TRUE);
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    SELECT * INTO reserve_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE experiment_hash = p_experiment_hash
       AND reservation_id = p_reservation_id
       AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC
     LIMIT 1;
    IF NOT FOUND
       OR reserve_event.binding_id IS DISTINCT FROM p_binding_id
       OR reserve_event.claim_key IS DISTINCT FROM p_claim_key
       OR reserve_event.claim_generation IS DISTINCT FROM p_claim_generation
       OR reserve_event.event_doc->>'reservation_id' IS DISTINCT FROM p_reservation_id
       OR reserve_event.event_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR reserve_event.event_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR reserve_event.event_doc->>'request_fingerprint' IS DISTINCT FROM p_request_fingerprint
       OR reserve_event.event_doc->>'action_id' IS DISTINCT FROM p_action_id
       OR reserve_event.event_doc->>'binding_catalog_manifest_hash'
            IS DISTINCT FROM p_binding_catalog_manifest_hash
       OR reserve_event.event_doc->>'call_grant_hash'
            IS DISTINCT FROM p_authorization_hash
       OR reserve_event.event_doc->>'request_body_hash'
            IS DISTINCT FROM p_request_body_hash
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_reservation_chain_mismatch' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO latest_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
     ORDER BY created_at DESC, event_key DESC
     LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_reservation_not_open' USING ERRCODE = '23505';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_routing_experiments_v2 experiment
        CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(experiment.spec_doc->'provider_bindings') AS binding(value)
        CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(experiment.spec_doc->'variants') AS variant(value)
        WHERE experiment.experiment_hash = p_experiment_hash
          AND experiment.receipt_execution_mode = p_execution_mode
          AND experiment.execution_envelope_doc->>'binding_catalog_manifest_hash'
                = p_binding_catalog_manifest_hash
          AND binding.value->>'binding_id' = p_binding_id
          AND binding.value->>'tool_id' = p_tool_id
          AND variant.value->>'variant_id' = p_variant_id
          AND p_binding_id = ANY (
              ARRAY(SELECT pg_catalog.jsonb_array_elements_text(variant.value->'binding_ids'))
          )
          AND EXISTS (
              SELECT 1
              FROM pg_catalog.jsonb_array_elements(
                  experiment.execution_envelope_doc->'bindings'
              ) AS runtime_binding(value)
              WHERE runtime_binding.value->>'binding_id' = p_binding_id
                AND runtime_binding.value->>'provider_id' = binding.value->>'provider_id'
                AND runtime_binding.value->>'tool_id' = p_tool_id
                AND runtime_binding.value->>'binding_manifest_hash'
                      = binding.value->>'manifest_hash'
                AND runtime_binding.value->>'action_id' = p_action_id
          )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_binding_not_declared' USING ERRCODE = '23503';
    END IF;
    IF p_attempt_doc->>'binding_id' IS DISTINCT FROM p_binding_id
       OR p_attempt_doc->>'tool_id' IS DISTINCT FROM p_tool_id
       OR p_attempt_doc->>'action_id' IS DISTINCT FROM p_action_id
       OR p_attempt_doc->>'binding_catalog_manifest_hash'
            IS DISTINCT FROM p_binding_catalog_manifest_hash
       OR p_attempt_doc->>'call_grant_hash' IS DISTINCT FROM p_authorization_hash
       OR p_attempt_doc->>'call_grant_proof_hash'
            IS DISTINCT FROM p_authorization_proof_hash
       OR p_attempt_doc->>'request_body_hash' IS DISTINCT FROM p_request_body_hash
       OR p_attempt_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR p_attempt_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR p_attempt_doc->>'reservation_id' IS DISTINCT FROM p_reservation_id
       OR p_attempt_doc->>'request_fingerprint' IS DISTINCT FROM p_request_fingerprint
       OR p_attempt_doc->>'execution_mode' IS DISTINCT FROM p_execution_mode
       OR public.research_lab_routing_jsonb_hash_v2(p_attempt_doc->'admission_bundle')
            IS DISTINCT FROM p_admission_bundle_hash
       OR p_attempt_doc->'admission_bundle'->>'job_id'
            IS DISTINCT FROM p_attempt_doc->'call_grant'->>'job_id'
       OR p_attempt_doc->'admission_bundle'->>'experiment_hash'
            IS DISTINCT FROM p_experiment_hash
       OR p_attempt_doc->'call_grant'->>'experiment_hash'
            IS DISTINCT FROM p_experiment_hash
       OR p_attempt_doc->'call_grant'->>'job_id'
            IS DISTINCT FROM p_attempt_doc->'terminal_proof'->'body'->>'job_id'
       OR p_attempt_doc->'call_grant'->>'admission_bundle_hash'
            IS DISTINCT FROM p_admission_bundle_hash
       OR p_attempt_doc->'call_grant'->>'binding_catalog_manifest_hash'
            IS DISTINCT FROM p_binding_catalog_manifest_hash
       OR p_attempt_doc->'call_grant'->>'action_id'
            IS DISTINCT FROM p_action_id
       OR p_attempt_doc->'call_grant'->>'unit_ref'
            IS DISTINCT FROM p_unit_ref
       OR p_attempt_doc->'call_grant'->>'core_request_fingerprint'
            IS DISTINCT FROM p_request_fingerprint
       OR p_attempt_doc->'call_grant'->>'request_body_hash'
            IS DISTINCT FROM p_request_body_hash
       OR p_attempt_doc->'call_grant'->'binding'->>'binding_id'
            IS DISTINCT FROM p_binding_id
       OR p_attempt_doc->'call_grant'->'binding'->>'tool_id'
            IS DISTINCT FROM p_tool_id
       OR p_attempt_doc->'call_grant_result'->>'authorization_hash'
            IS DISTINCT FROM p_authorization_hash
       OR p_attempt_doc->'call_grant_result'->>'job_id'
            IS DISTINCT FROM p_attempt_doc->'call_grant'->>'job_id'
       OR p_attempt_doc->'call_grant_result'->>'experiment_hash'
            IS DISTINCT FROM p_attempt_doc->'call_grant'->>'experiment_hash'
       OR p_attempt_doc->'call_grant_receipt'->>'receipt_hash'
            IS DISTINCT FROM p_authorization_proof_hash
       OR p_attempt_doc->'call_grant_receipt'->>'input_root'
            IS DISTINCT FROM p_authorization_hash
       OR p_attempt_doc->'call_grant_receipt'->>'output_root'
            IS DISTINCT FROM p_attempt_doc->'call_grant_result'->>'output_root'
       OR p_attempt_doc->'terminal_proof'->'body'->>'experiment_hash'
            IS DISTINCT FROM p_experiment_hash
       OR p_attempt_doc->'terminal_proof'->'body'->>'admission_bundle_hash'
            IS DISTINCT FROM p_admission_bundle_hash
       OR p_attempt_doc->'terminal_proof'->'body'->>'authorization_hash'
            IS DISTINCT FROM p_authorization_hash
       OR p_attempt_doc->'terminal_proof'->'body'->>'authorization_proof_hash'
            IS DISTINCT FROM p_authorization_proof_hash
       OR p_attempt_doc->'terminal_proof'->'body'->'binding'->>'binding_id'
            IS DISTINCT FROM p_binding_id
       OR p_attempt_doc->'terminal_proof'->'body'->'binding'->>'tool_id'
            IS DISTINCT FROM p_tool_id
       OR p_attempt_doc->'terminal_proof'->'body'->>'variant_id'
            IS DISTINCT FROM p_variant_id
       OR p_attempt_doc->'terminal_proof'->'body'->>'unit_ref'
            IS DISTINCT FROM p_unit_ref
       OR p_attempt_doc->'terminal_proof'->'body'->>'request_fingerprint'
            IS DISTINCT FROM p_request_fingerprint
       OR p_attempt_doc->'terminal_proof'->'receipt'->>'receipt_hash'
            IS DISTINCT FROM p_terminal_receipt_hash
       OR p_attempt_doc->'terminal_proof'->'body'->>'provider_record_hash'
            IS DISTINCT FROM p_terminal_provider_record_hash
       OR p_attempt_doc->'terminal_proof'->'body'->>'billing_projection_hash'
            IS DISTINCT FROM p_terminal_billing_projection_hash
       OR p_attempt_doc->'protected_release_receipt'->>'receipt_hash'
            IS DISTINCT FROM p_protected_release_receipt_hash
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_document_mismatch' USING ERRCODE = '22023';
    END IF;
    billing_projection := COALESCE(
        p_attempt_doc->'terminal_proof'->'body'->'billing_projection',
        p_attempt_doc->'terminal_proof'->'projection'
    );
    IF pg_catalog.jsonb_typeof(billing_projection) <> 'object'
       OR billing_projection->>'outcome' IS DISTINCT FROM p_outcome
       OR billing_projection->>'credit_microunits' IS NULL
       OR billing_projection->>'credit_microunits' !~ '^[0-9]+$'
       OR (billing_projection->>'credit_microunits')::BIGINT IS DISTINCT FROM p_credit_microunits
       OR billing_projection->>'latency_ms' IS NULL
       OR billing_projection->>'latency_ms' !~ '^[0-9]+$'
       OR (billing_projection->>'latency_ms')::BIGINT IS DISTINCT FROM p_latency_ms
       OR billing_projection->>'billing_state' IS DISTINCT FROM 'known'
       OR p_billing_state IS DISTINCT FROM 'known'
       OR p_authoritative_billed_credit_microunits IS DISTINCT FROM p_credit_microunits
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_authoritative_billing_mismatch'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
        p_experiment_hash, p_binding_id, p_tool_id, p_variant_id, p_unit_ref,
        p_action_id, p_authorization_hash, p_authorization_proof_hash,
        p_terminal_receipt_hash, p_protected_release_receipt_hash,
        p_admission_bundle_hash, p_terminal_provider_record_hash,
        p_terminal_billing_projection_hash, p_outcome, p_credit_microunits,
        p_latency_ms, p_billing_state,
        p_authoritative_billed_credit_microunits, p_attempt_doc
    );
    INSERT INTO public.research_lab_routing_provider_attempts_v2 (
        attempt_key, experiment_hash, provider_receipt_ref, binding_id, tool_id,
        variant_id, unit_ref, reservation_id, action_id, claim_key, claim_generation,
        binding_catalog_manifest_hash, authorization_hash,
        authorization_proof_hash, request_body_hash,
        terminal_receipt_hash, protected_release_receipt_hash,
        admission_bundle_hash, terminal_provider_record_hash,
        terminal_billing_projection_hash,
        request_fingerprint, outcome, credit_microunits, latency_ms,
        execution_mode, billing_state, authoritative_billed_credit_microunits,
        attempt_doc
    ) VALUES (
        p_attempt_key, p_experiment_hash, p_provider_receipt_ref, p_binding_id,
        p_tool_id, p_variant_id, p_unit_ref, p_reservation_id, p_action_id,
        p_claim_key, p_claim_generation,
        p_binding_catalog_manifest_hash, p_authorization_hash,
        p_authorization_proof_hash, p_request_body_hash,
        p_terminal_receipt_hash, p_protected_release_receipt_hash,
        p_admission_bundle_hash, p_terminal_provider_record_hash,
        p_terminal_billing_projection_hash,
        p_request_fingerprint, p_outcome, p_credit_microunits,
        p_latency_ms, p_execution_mode, p_billing_state,
        p_authoritative_billed_credit_microunits, p_attempt_doc
    );
    RETURN pg_catalog.jsonb_build_object('attempt_key', p_attempt_key, 'idempotent', FALSE);
END;
$attempt$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_decision_receipt_v2(
    p_receipt_id TEXT,
    p_experiment_hash TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_plan_hash TEXT,
    p_route_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_decision_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $decision$
DECLARE
    existing public.research_lab_routing_decision_receipts_v2%ROWTYPE;
BEGIN
    IF p_receipt_id !~ '^routing_decision:[0-9a-f]{16}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_unit_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_plan_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_route_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_decision_doc->>'receipt_id' IS DISTINCT FROM p_receipt_id
    THEN
        RAISE EXCEPTION 'research_lab_routing_decision_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_decision_doc, 'routing decision');
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_routing_experiments_v2 experiment
        CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(experiment.spec_doc->'variants') AS variant(value)
        WHERE experiment.experiment_hash = p_experiment_hash
          AND variant.value->>'variant_id' = p_variant_id
    ) OR p_decision_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR p_decision_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR p_decision_doc->>'plan_hash' IS DISTINCT FROM p_plan_hash
       OR p_decision_doc->>'route_hash' IS DISTINCT FROM p_route_hash
    THEN
        RAISE EXCEPTION 'research_lab_routing_decision_spec_or_document_mismatch' USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_receipt_id, 0));
    SELECT * INTO existing FROM public.research_lab_routing_decision_receipts_v2 WHERE receipt_id = p_receipt_id;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.variant_id IS DISTINCT FROM p_variant_id
           OR existing.unit_ref IS DISTINCT FROM p_unit_ref
           OR existing.plan_hash IS DISTINCT FROM p_plan_hash
           OR existing.route_hash IS DISTINCT FROM p_route_hash
           OR existing.decision_doc IS DISTINCT FROM p_decision_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_decision_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('receipt_id', p_receipt_id, 'idempotent', TRUE);
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_decision_doc->'provider_receipt_refs', '[]'::JSONB)
        ) AS ref(provider_receipt_ref)
        WHERE NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_provider_attempts_v2 attempt
             WHERE attempt.experiment_hash = p_experiment_hash
               AND attempt.provider_receipt_ref = ref.provider_receipt_ref
        )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_decision_provider_receipt_missing' USING ERRCODE = '23503';
    END IF;
    INSERT INTO public.research_lab_routing_decision_receipts_v2 (
        receipt_id, experiment_hash, variant_id, unit_ref, claim_key,
        claim_generation, plan_hash, route_hash, decision_doc
    ) VALUES (
        p_receipt_id, p_experiment_hash, p_variant_id, p_unit_ref, p_claim_key,
        p_claim_generation, p_plan_hash, p_route_hash, p_decision_doc
    );
    RETURN pg_catalog.jsonb_build_object('receipt_id', p_receipt_id, 'idempotent', FALSE);
END;
$decision$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_evaluation_v2(
    p_receipt_id TEXT,
    p_experiment_hash TEXT,
    p_evaluation_hash TEXT,
    p_selected_variant_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_evaluation_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $evaluation$
DECLARE
    existing public.research_lab_routing_evaluation_receipts_v2%ROWTYPE;
    experiment public.research_lab_routing_experiments_v2%ROWTYPE;
BEGIN
    IF p_receipt_id !~ '^routing_evaluation_v2:[0-9a-f]{16}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_evaluation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR (p_selected_variant_id <> 'unselected'
           AND p_selected_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$')
       OR p_evaluation_doc->>'receipt_id' IS DISTINCT FROM p_receipt_id
       OR p_evaluation_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR (
           p_evaluation_doc->>'selected_variant_id' IS DISTINCT FROM p_selected_variant_id
           AND NOT (
               p_selected_variant_id = 'unselected'
               AND p_evaluation_doc->>'selected_variant_id' = ''
           )
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_evaluation_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_evaluation_doc, 'routing evaluation');
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_receipt_id, 0));
    SELECT * INTO experiment
      FROM public.research_lab_routing_experiments_v2
     WHERE experiment_hash = p_experiment_hash;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_evaluation_experiment_missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing FROM public.research_lab_routing_evaluation_receipts_v2 WHERE receipt_id = p_receipt_id;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.evaluation_hash IS DISTINCT FROM p_evaluation_hash
           OR existing.selected_variant_id IS DISTINCT FROM p_selected_variant_id
           OR existing.evaluation_doc IS DISTINCT FROM p_evaluation_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_evaluation_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('receipt_id', p_receipt_id, 'idempotent', TRUE);
    END IF;
    IF pg_catalog.jsonb_array_length(
        coalesce(p_evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
    ) = 0
       OR (
        p_selected_variant_id <> 'unselected'
        AND (
            NOT EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(
                    experiment.spec_doc->'variants'
                ) AS variant(value)
                WHERE variant.value->>'variant_id' = p_selected_variant_id
            )
            OR NOT EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(
                    coalesce(p_evaluation_doc->'variants', '[]'::JSONB)
                ) AS variant(value)
                WHERE variant.value->>'variant_id' = p_selected_variant_id
            )
        )
       )
       OR pg_catalog.jsonb_array_length(
        coalesce(p_evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
    ) = 0
       OR EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
        ) AS ref(receipt_id)
        WHERE NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_decision_receipts_v2 decision
             WHERE decision.experiment_hash = p_experiment_hash
               AND decision.receipt_id = ref.receipt_id
        )
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
        ) AS ref(provider_receipt_ref)
        WHERE NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_provider_attempts_v2 attempt
             WHERE attempt.experiment_hash = p_experiment_hash
               AND attempt.provider_receipt_ref = ref.provider_receipt_ref
        )
    ) OR EXISTS (
        (SELECT decision.receipt_id
           FROM public.research_lab_routing_decision_receipts_v2 decision
          WHERE decision.experiment_hash = p_experiment_hash)
        EXCEPT
        (SELECT ref.receipt_id
           FROM pg_catalog.jsonb_array_elements_text(
               coalesce(p_evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
           ) AS ref(receipt_id))
    ) OR EXISTS (
        (SELECT attempt.provider_receipt_ref
           FROM public.research_lab_routing_provider_attempts_v2 attempt
          WHERE attempt.experiment_hash = p_experiment_hash)
        EXCEPT
        (SELECT ref.provider_receipt_ref
           FROM pg_catalog.jsonb_array_elements_text(
               coalesce(p_evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
           ) AS ref(provider_receipt_ref))
    ) OR (
        SELECT count(*)
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
        ) AS ref(receipt_id)
    ) <> (
        SELECT count(DISTINCT ref.receipt_id)
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
        ) AS ref(receipt_id)
    ) OR (
        SELECT count(*)
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
        ) AS ref(provider_receipt_ref)
    ) <> (
        SELECT count(DISTINCT ref.provider_receipt_ref)
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(p_evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
        ) AS ref(provider_receipt_ref)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_evaluation_receipt_set_is_not_complete' USING ERRCODE = '23503';
    END IF;
    INSERT INTO public.research_lab_routing_evaluation_receipts_v2 (
        receipt_id, experiment_hash, evaluation_hash, selected_variant_id,
        claim_key, claim_generation, evaluation_doc
    ) VALUES (
        p_receipt_id, p_experiment_hash, p_evaluation_hash, p_selected_variant_id,
        p_claim_key, p_claim_generation, p_evaluation_doc
    );
    RETURN pg_catalog.jsonb_build_object('receipt_id', p_receipt_id, 'idempotent', FALSE);
END;
$evaluation$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_reserve_budget_v2(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_experiment_hash TEXT,
    p_binding_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_credit_microunits BIGINT,
    p_lease_seconds INTEGER,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $reserve$
DECLARE
    existing public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    experiment public.research_lab_routing_experiments_v2%ROWTYPE;
    total_budget BIGINT;
    binding_budget BIGINT;
    consumed BIGINT;
    binding_consumed BIGINT;
    expiry TIMESTAMPTZ;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_binding_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_credit_microunits < 0 OR p_credit_microunits > 10000000
       OR p_lease_seconds < 1 OR p_lease_seconds > 3600
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing budget reserve');
    IF p_event_doc->>'reservation_id' IS DISTINCT FROM p_reservation_id
       OR p_event_doc->>'binding_id' IS DISTINCT FROM p_binding_id
       OR p_event_doc->>'unit_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'variant_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'request_fingerprint' !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_doc->>'action_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'tool_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'binding_catalog_manifest_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_doc->>'call_grant_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_doc->>'request_body_hash' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_chain_identity_missing' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    SELECT * INTO experiment FROM public.research_lab_routing_experiments_v2 WHERE experiment_hash = p_experiment_hash;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_experiment_missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing FROM public.research_lab_routing_budget_events_v2 WHERE event_key = p_event_key;
    IF FOUND THEN
        IF existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.binding_id IS DISTINCT FROM p_binding_id
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_type IS DISTINCT FROM 'reserve'
           OR existing.credit_microunits IS DISTINCT FROM p_credit_microunits
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_reserve_conflict' USING ERRCODE = '23505';
        END IF;
        SELECT * INTO latest_event FROM public.research_lab_routing_budget_events_v2
         WHERE reservation_id = p_reservation_id
         ORDER BY created_at DESC, event_key DESC LIMIT 1;
        IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
            RAISE EXCEPTION 'research_lab_routing_reserve_is_closed' USING ERRCODE = '23505';
        END IF;
        IF existing.lease_expires_at <= pg_catalog.clock_timestamp() THEN
            RAISE EXCEPTION 'research_lab_routing_reserve_is_expired' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('reserved', TRUE, 'idempotent', TRUE, 'lease_expires_at', existing.lease_expires_at);
    END IF;
    total_budget := coalesce((experiment.spec_doc #>> '{credit_budget,total_credit_microunits}')::BIGINT, -1);
    binding_budget := coalesce((experiment.spec_doc #>> ARRAY['credit_budget', 'provider_credit_ceilings', p_binding_id])::BIGINT, -1);
    IF experiment.allow_live_credit_spend IS NOT TRUE
       OR experiment.receipt_execution_mode IS DISTINCT FROM 'measured_lab'
       OR total_budget < 0
       OR binding_budget < 0
       OR NOT EXISTS (
           SELECT 1
           FROM pg_catalog.jsonb_array_elements(
                    experiment.spec_doc->'provider_bindings'
                ) AS model_binding(value)
           JOIN pg_catalog.jsonb_array_elements(
                    experiment.execution_envelope_doc->'bindings'
                ) AS runtime_binding(value)
             ON runtime_binding.value->>'binding_id'
                    = model_binding.value->>'binding_id'
           WHERE model_binding.value->>'binding_id' = p_binding_id
             AND model_binding.value->>'tool_id'
                    = p_event_doc->>'tool_id'
             AND runtime_binding.value->>'tool_id'
                    = p_event_doc->>'tool_id'
             AND runtime_binding.value->>'provider_id'
                    = model_binding.value->>'provider_id'
             AND runtime_binding.value->>'binding_manifest_hash'
                    = model_binding.value->>'manifest_hash'
             AND runtime_binding.value->>'action_id'
                    = p_event_doc->>'action_id'
             AND experiment.execution_envelope_doc->>'binding_catalog_manifest_hash'
                    = p_event_doc->>'binding_catalog_manifest_hash'
             AND coalesce(
                    (runtime_binding.value->>'credit_ceiling_microunits')::BIGINT,
                    -1
                 ) >= p_credit_microunits
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_budget_contract_missing' USING ERRCODE = '22023';
    END IF;
    SELECT coalesce(sum(head.credit_microunits), 0) INTO consumed
    FROM (
        SELECT DISTINCT ON (reservation_id) *
        FROM public.research_lab_routing_budget_events_v2
        WHERE experiment_hash = p_experiment_hash
        ORDER BY reservation_id, created_at DESC, event_key DESC
    ) AS head
    WHERE head.event_type IN ('settle', 'uncertain', 'recover')
       OR (head.event_type = 'reserve' AND head.lease_expires_at > pg_catalog.clock_timestamp());
    SELECT coalesce(sum(head.credit_microunits), 0) INTO binding_consumed
    FROM (
        SELECT DISTINCT ON (reservation_id) *
        FROM public.research_lab_routing_budget_events_v2
        WHERE experiment_hash = p_experiment_hash AND binding_id = p_binding_id
        ORDER BY reservation_id, created_at DESC, event_key DESC
    ) AS head
    WHERE head.event_type IN ('settle', 'uncertain', 'recover')
       OR (head.event_type = 'reserve' AND head.lease_expires_at > pg_catalog.clock_timestamp());
    IF consumed + p_credit_microunits > total_budget
       OR binding_consumed + p_credit_microunits > binding_budget
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_budget_exceeded' USING ERRCODE = '23505';
    END IF;
    expiry := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_seconds);
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type,
        credit_microunits, lease_expires_at, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, p_experiment_hash, p_binding_id,
        p_claim_key, p_claim_generation, 'reserve', p_credit_microunits,
        expiry, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object('reserved', TRUE, 'idempotent', FALSE, 'lease_expires_at', expiry);
END;
$reserve$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_settle_budget_v2(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_attempt_key TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $settle$
DECLARE
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    attempt public.research_lab_routing_provider_attempts_v2%ROWTYPE;
    existing public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_attempt_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_settle_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing budget settlement');
    SELECT * INTO reserve_event FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_settle_reservation_missing' USING ERRCODE = '23503';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        reserve_event.experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    SELECT * INTO existing FROM public.research_lab_routing_budget_events_v2 WHERE event_key = p_event_key;
    IF FOUND THEN
        IF existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
           OR existing.binding_id IS DISTINCT FROM reserve_event.binding_id
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.attempt_key IS DISTINCT FROM p_attempt_key
           OR existing.event_type IS DISTINCT FROM 'settle'
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_settle_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('settled', TRUE, 'idempotent', TRUE);
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(reserve_event.experiment_hash, 0));
    SELECT * INTO latest_event FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id ORDER BY created_at DESC, event_key DESC LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
        RAISE EXCEPTION 'research_lab_routing_settle_reservation_closed' USING ERRCODE = '23505';
    END IF;
    SELECT * INTO attempt FROM public.research_lab_routing_provider_attempts_v2 WHERE attempt_key = p_attempt_key;
    IF NOT FOUND
       OR attempt.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
       OR attempt.binding_id IS DISTINCT FROM reserve_event.binding_id
       OR attempt.reservation_id IS DISTINCT FROM p_reservation_id
       OR attempt.unit_ref IS DISTINCT FROM reserve_event.event_doc->>'unit_ref'
       OR attempt.variant_id IS DISTINCT FROM reserve_event.event_doc->>'variant_id'
       OR attempt.request_fingerprint IS DISTINCT FROM reserve_event.event_doc->>'request_fingerprint'
       OR attempt.action_id IS DISTINCT FROM reserve_event.event_doc->>'action_id'
       OR attempt.claim_key IS DISTINCT FROM p_claim_key
       OR attempt.claim_generation IS DISTINCT FROM p_claim_generation
       OR attempt.billing_state IS DISTINCT FROM 'known'
       OR attempt.authoritative_billed_credit_microunits > reserve_event.credit_microunits
    THEN
        RAISE EXCEPTION 'research_lab_routing_settle_attempt_mismatch' USING ERRCODE = '23503';
    END IF;
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, attempt_key,
        event_type, credit_microunits, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, reserve_event.experiment_hash,
        reserve_event.binding_id, p_claim_key, p_claim_generation, p_attempt_key,
        'settle', attempt.authoritative_billed_credit_microunits, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object('settled', TRUE, 'idempotent', FALSE, 'credit_microunits', attempt.authoritative_billed_credit_microunits);
END;
$settle$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_mark_budget_uncertain_v2(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $uncertain$
DECLARE
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing uncertain budget');
    SELECT * INTO reserve_event FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_reservation_missing' USING ERRCODE = '23503';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        reserve_event.experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_budget_events_v2 existing
        WHERE existing.event_key = p_event_key
          AND (
              existing.reservation_id IS DISTINCT FROM p_reservation_id
              OR existing.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
              OR existing.binding_id IS DISTINCT FROM reserve_event.binding_id
              OR existing.claim_key IS DISTINCT FROM p_claim_key
              OR existing.claim_generation IS DISTINCT FROM p_claim_generation
              OR existing.event_type IS DISTINCT FROM 'uncertain'
              OR existing.credit_microunits IS DISTINCT FROM reserve_event.credit_microunits
              OR existing.event_doc IS DISTINCT FROM p_event_doc
          )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_conflict' USING ERRCODE = '23505';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_budget_events_v2
        WHERE event_key = p_event_key
    ) THEN
        RETURN pg_catalog.jsonb_build_object('uncertain', TRUE, 'idempotent', TRUE);
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(reserve_event.experiment_hash, 0));
    SELECT * INTO latest_event FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id ORDER BY created_at DESC, event_key DESC LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_reservation_closed' USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type,
        credit_microunits, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, reserve_event.experiment_hash,
        reserve_event.binding_id, p_claim_key, p_claim_generation, 'uncertain',
        reserve_event.credit_microunits, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object('uncertain', TRUE, 'idempotent', FALSE);
END;
$uncertain$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_recover_budget_v2(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $recover_budget$
DECLARE
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing budget recovery');
    SELECT * INTO reserve_event FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_reservation_missing' USING ERRCODE = '23503';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        reserve_event.experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_budget_events_v2 existing
        WHERE existing.event_key = p_event_key
          AND (
              existing.reservation_id IS DISTINCT FROM p_reservation_id
              OR existing.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
              OR existing.binding_id IS DISTINCT FROM reserve_event.binding_id
              OR existing.claim_key IS DISTINCT FROM p_claim_key
              OR existing.claim_generation IS DISTINCT FROM p_claim_generation
              OR existing.event_type IS DISTINCT FROM 'recover'
              OR existing.credit_microunits IS DISTINCT FROM reserve_event.credit_microunits
              OR existing.event_doc IS DISTINCT FROM p_event_doc
          )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_conflict' USING ERRCODE = '23505';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_budget_events_v2
        WHERE event_key = p_event_key
    ) THEN
        RETURN pg_catalog.jsonb_build_object('recovered', TRUE, 'idempotent', TRUE);
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(reserve_event.experiment_hash, 0));
    SELECT * INTO latest_event FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id ORDER BY created_at DESC, event_key DESC LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve'
       OR latest_event.lease_expires_at >= pg_catalog.clock_timestamp()
    THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_not_expired' USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type,
        credit_microunits, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, reserve_event.experiment_hash,
        reserve_event.binding_id, p_claim_key, p_claim_generation, 'recover',
        reserve_event.credit_microunits, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'recovered', TRUE,
        'idempotent', FALSE,
        'billing_state', 'uncertain',
        'credit_microunits', reserve_event.credit_microunits
    );
END;
$recover_budget$;

-- A resumed worker must conservatively close every expired open reservation
-- before it starts another provider call.  An absent dispatch marker is not
-- sufficient evidence that a transport did not start, so callers must treat
-- every returned reservation as full-ceiling unknown billing.
CREATE OR REPLACE FUNCTION public.research_lab_routing_list_expired_budget_reservations_v2(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $list_expired_budgets$
DECLARE
    reservations JSONB;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation < 1
    THEN
        RAISE EXCEPTION 'research_lab_routing_list_expired_budgets_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    WITH latest AS (
        SELECT DISTINCT ON (event.reservation_id)
            event.reservation_id,
            event.binding_id,
            event.claim_key AS reserve_claim_key,
            event.claim_generation AS reserve_claim_generation,
            event.credit_microunits,
            event.event_type,
            event.lease_expires_at
        FROM public.research_lab_routing_budget_events_v2 event
        WHERE event.experiment_hash = p_experiment_hash
        ORDER BY event.reservation_id, event.created_at DESC, event.event_key DESC
    ), expired AS (
        SELECT *
        FROM latest
        WHERE event_type = 'reserve'
          AND lease_expires_at <= pg_catalog.clock_timestamp()
    )
    SELECT coalesce(
        pg_catalog.jsonb_agg(
            pg_catalog.jsonb_build_object(
                'reservation_id', expired.reservation_id,
                'binding_id', expired.binding_id,
                'credit_microunits', expired.credit_microunits,
                'dispatch_started', EXISTS (
                    SELECT 1
                    FROM public.research_lab_routing_experiment_events_v2 dispatch_event
                    WHERE dispatch_event.experiment_hash = p_experiment_hash
                      AND dispatch_event.event_type = 'provider_dispatch_started'
                      AND dispatch_event.claim_key = expired.reserve_claim_key
                      AND dispatch_event.claim_generation = expired.reserve_claim_generation
                      AND dispatch_event.event_doc->>'reservation_id' = expired.reservation_id
                      AND dispatch_event.event_doc->>'binding_id' = expired.binding_id
                )
            ) ORDER BY expired.reservation_id
        ), '[]'::JSONB
    ) INTO reservations
    FROM expired;
    RETURN pg_catalog.jsonb_build_object('reservations', reservations);
END;
$list_expired_budgets$;

-- Every non-settled budget head blocks a resumed provider run.  This is kept
-- separate from the expired-only helper above so an older `uncertain` head
-- cannot disappear from a later recovery scan.  The result is fully
-- canonical and contains redacted ledger data only.
CREATE OR REPLACE FUNCTION public.research_lab_routing_list_unresolved_budget_reservations_v2(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_claim_nonce TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $list_unresolved_budgets$
DECLARE
    reservations JSONB;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation < 1
    THEN
        RAISE EXCEPTION 'research_lab_routing_list_unresolved_budgets_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v2(
        p_experiment_hash, p_claim_key, p_claim_generation, p_claim_nonce
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    WITH latest AS (
        SELECT DISTINCT ON (event.reservation_id)
            event.reservation_id,
            event.binding_id,
            event.claim_key AS reserve_claim_key,
            event.claim_generation AS reserve_claim_generation,
            event.credit_microunits,
            event.event_type,
            event.lease_expires_at
        FROM public.research_lab_routing_budget_events_v2 event
        WHERE event.experiment_hash = p_experiment_hash
        ORDER BY event.reservation_id, event.created_at DESC, event.event_key DESC
    ), unresolved AS (
        SELECT * FROM latest WHERE event_type <> 'settle'
    )
    SELECT coalesce(
        pg_catalog.jsonb_agg(
            pg_catalog.jsonb_build_object(
                'reservation_id', unresolved.reservation_id,
                'binding_id', unresolved.binding_id,
                'credit_microunits', unresolved.credit_microunits,
                'event_type', unresolved.event_type,
                'lease_expired', (
                    unresolved.event_type = 'reserve'
                    AND unresolved.lease_expires_at <= pg_catalog.clock_timestamp()
                ),
                'dispatch_started', EXISTS (
                    SELECT 1
                    FROM public.research_lab_routing_experiment_events_v2 dispatch_event
                    WHERE dispatch_event.experiment_hash = p_experiment_hash
                      AND dispatch_event.event_type = 'provider_dispatch_started'
                      AND dispatch_event.claim_key = unresolved.reserve_claim_key
                      AND dispatch_event.claim_generation = unresolved.reserve_claim_generation
                      AND dispatch_event.event_doc->>'reservation_id' = unresolved.reservation_id
                      AND dispatch_event.event_doc->>'binding_id' = unresolved.binding_id
                )
            ) ORDER BY unresolved.reservation_id
        ), '[]'::JSONB
    ) INTO reservations
    FROM unresolved;
    RETURN pg_catalog.jsonb_build_object('reservations', reservations);
END;
$list_unresolved_budgets$;

-- Promotion repeats the receipt-chain check over every durable attempt.  A
-- caller must not be able to validate a clean evaluation once and then alter
-- or substitute one attempt before the Lab reference is written.
CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_promotion_receipt_chain_v2(
    p_experiment_hash TEXT
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $promotion_chain$
DECLARE
    attempt_row public.research_lab_routing_provider_attempts_v2%ROWTYPE;
BEGIN
    -- Reuse the append-time validator so promotion observes exactly the
    -- same standard receipt graph and signer/billing bindings.
    FOR attempt_row IN
        SELECT *
        FROM public.research_lab_routing_provider_attempts_v2
        WHERE experiment_hash = p_experiment_hash
        ORDER BY attempt_key
    LOOP
        IF attempt_row.outcome = 'adapter_failure'
           OR attempt_row.billing_state IS DISTINCT FROM 'known'
           OR attempt_row.authoritative_billed_credit_microunits
                IS DISTINCT FROM attempt_row.credit_microunits
           OR attempt_row.terminal_receipt_hash IS NULL
           OR attempt_row.protected_release_receipt_hash IS NULL
           OR attempt_row.admission_bundle_hash IS NULL
           OR attempt_row.terminal_provider_record_hash IS NULL
           OR attempt_row.terminal_billing_projection_hash IS NULL
        THEN
            RAISE EXCEPTION 'research_lab_routing_promote_receipt_chain_missing'
                USING ERRCODE = '23503';
        END IF;
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v2(
            attempt_row.experiment_hash, attempt_row.binding_id,
            attempt_row.tool_id, attempt_row.variant_id, attempt_row.unit_ref,
            attempt_row.action_id, attempt_row.authorization_hash,
            attempt_row.authorization_proof_hash,
            attempt_row.terminal_receipt_hash,
            attempt_row.protected_release_receipt_hash,
            attempt_row.admission_bundle_hash,
            attempt_row.terminal_provider_record_hash,
            attempt_row.terminal_billing_projection_hash,
            attempt_row.outcome, attempt_row.credit_microunits,
            attempt_row.latency_ms, attempt_row.billing_state,
            attempt_row.authoritative_billed_credit_microunits,
            attempt_row.attempt_doc
        );
    END LOOP;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_routing_provider_attempts_v2 attempt
        LEFT JOIN public.research_lab_attested_execution_receipts_v2 authorization_receipt
          ON authorization_receipt.receipt_hash = attempt.authorization_proof_hash
         AND authorization_receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
         AND authorization_receipt.role = 'gateway_scoring'
         AND authorization_receipt.purpose = 'research_lab.routing_provider_evidence.v2'
         AND authorization_receipt.receipt_status = 'succeeded'
         AND authorization_receipt.job_id = attempt.attempt_doc->'call_grant'->>'job_id'
         AND authorization_receipt.input_root = attempt.authorization_hash
         AND authorization_receipt.output_root = attempt.attempt_doc->'call_grant_result'->>'output_root'
         AND authorization_receipt.receipt_doc = attempt.attempt_doc->'call_grant_receipt'
         AND authorization_receipt.receipt_doc->'parent_receipt_hashes' = '[]'::JSONB
        LEFT JOIN public.research_lab_attested_execution_receipts_v2 protected_receipt
          ON protected_receipt.receipt_hash = attempt.protected_release_receipt_hash
         AND protected_receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
         AND protected_receipt.role = 'gateway_scoring'
         AND protected_receipt.purpose = 'research_lab.routing_provider_evidence.v2'
         AND protected_receipt.receipt_status = 'succeeded'
         AND protected_receipt.job_id = attempt.attempt_doc->'admission_bundle'->>'job_id'
         AND protected_receipt.receipt_doc = attempt.attempt_doc->'protected_release_receipt'
         AND protected_receipt.receipt_doc->'parent_receipt_hashes' = '[]'::JSONB
        LEFT JOIN public.research_lab_attested_execution_receipts_v2 terminal_receipt
          ON terminal_receipt.receipt_hash = attempt.terminal_receipt_hash
         AND terminal_receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
         AND terminal_receipt.role = 'gateway_scoring'
         AND terminal_receipt.purpose = 'research_lab.routing_provider_evidence.v2'
         AND terminal_receipt.receipt_status = 'succeeded'
         AND terminal_receipt.job_id = attempt.attempt_doc->'terminal_proof'->'body'->>'job_id'
         AND terminal_receipt.input_root = public.research_lab_routing_jsonb_hash_v2(
             attempt.attempt_doc->'terminal_proof'->'body'
         )
         AND terminal_receipt.output_root = attempt.attempt_doc->'terminal_proof'->'receipt'->>'output_root'
         AND terminal_receipt.transport_root = attempt.terminal_provider_record_hash
         AND terminal_receipt.host_operation_root = attempt.authorization_hash
         AND terminal_receipt.artifact_root = attempt.terminal_billing_projection_hash
         AND terminal_receipt.receipt_doc = attempt.attempt_doc->'terminal_proof'->'receipt'
         AND terminal_receipt.receipt_doc->'parent_receipt_hashes'
             = pg_catalog.jsonb_build_array(attempt.protected_release_receipt_hash)
        WHERE attempt.experiment_hash = p_experiment_hash
          AND (
              attempt.outcome = 'adapter_failure'
              OR attempt.billing_state <> 'known'
              OR attempt.authoritative_billed_credit_microunits IS DISTINCT FROM attempt.credit_microunits
              OR attempt.admission_bundle_hash IS NULL
              OR attempt.terminal_receipt_hash IS NULL
              OR attempt.protected_release_receipt_hash IS NULL
              OR attempt.terminal_provider_record_hash IS NULL
              OR attempt.terminal_billing_projection_hash IS NULL
              OR authorization_receipt.receipt_hash IS NULL
              OR protected_receipt.receipt_hash IS NULL
              OR terminal_receipt.receipt_hash IS NULL
          )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_promote_receipt_chain_missing' USING ERRCODE = '23503';
    END IF;
END;
$promotion_chain$;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_promotion_receipt_chain_v2(TEXT)
    FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION public.research_lab_routing_promote_v2(
    p_reference_hash TEXT,
    p_experiment_hash TEXT,
    p_evaluation_receipt_id TEXT,
    p_evaluation_hash TEXT,
    p_selected_variant_id TEXT,
    p_reconciliation_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $promote$
DECLARE
    evaluation public.research_lab_routing_evaluation_receipts_v2%ROWTYPE;
    existing public.research_lab_routing_lab_references_v2%ROWTYPE;
BEGIN
    IF p_reference_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_evaluation_receipt_id !~ '^routing_evaluation_v2:[0-9a-f]{16}$'
       OR p_evaluation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_selected_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_reconciliation_doc, 'routing reconciliation');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing promotion event');
    PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended(p_experiment_hash, 0));
    -- Promotion must re-run the same standard receipt-graph validator used
    -- at append time.  A previously clean evaluation is not proof that an
    -- attempt still points at the same job, signer, parents, or bill.
    PERFORM public.research_lab_routing_assert_promotion_receipt_chain_v2(
        p_experiment_hash
    );
    SELECT * INTO evaluation FROM public.research_lab_routing_evaluation_receipts_v2
     WHERE receipt_id = p_evaluation_receipt_id;
    IF NOT FOUND
       OR evaluation.experiment_hash IS DISTINCT FROM p_experiment_hash
       OR evaluation.evaluation_hash IS DISTINCT FROM p_evaluation_hash
       OR evaluation.selected_variant_id IS DISTINCT FROM p_selected_variant_id
       OR p_reconciliation_doc->>'evaluation_receipt_id' IS DISTINCT FROM p_evaluation_receipt_id
       OR p_reconciliation_doc->>'evaluation_hash' IS DISTINCT FROM p_evaluation_hash
       OR p_reconciliation_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR p_reconciliation_doc->>'selected_variant_id' IS DISTINCT FROM p_selected_variant_id
       OR (p_reconciliation_doc->>'reconciled')::BOOLEAN IS DISTINCT FROM TRUE
       OR p_reconciliation_doc->>'authority_receipt_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'authority_input_root' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'authority_output_root' !~ '^sha256:[0-9a-f]{64}$'
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_execution_receipts_v2 authority_receipt
           WHERE authority_receipt.receipt_hash = p_reconciliation_doc->>'authority_receipt_hash'
             AND authority_receipt.role = 'gateway_scoring'
             AND authority_receipt.purpose = 'research_lab.routing_experiment.v2'
             AND authority_receipt.receipt_status = 'succeeded'
             AND authority_receipt.input_root = p_reconciliation_doc->>'authority_input_root'
             AND authority_receipt.output_root = p_reconciliation_doc->>'authority_output_root'
             AND authority_receipt.commit_sha = p_reconciliation_doc->>'authority_commit_sha'
             AND authority_receipt.pcr0 = p_reconciliation_doc->>'authority_pcr0'
             AND authority_receipt.build_manifest_hash
                    = p_reconciliation_doc->>'authority_build_manifest_hash'
             AND authority_receipt.boot_identity_hash
                    = p_reconciliation_doc->>'authority_boot_identity_hash'
             AND authority_receipt.receipt_doc->>'receipt_hash'
                    = p_reconciliation_doc->>'authority_receipt_hash'
       )
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_routing_experiments_v2 experiment
           WHERE experiment.experiment_hash = p_experiment_hash
             AND experiment.execution_envelope_hash
                    = p_reconciliation_doc->>'execution_envelope_hash'
             AND experiment.execution_envelope_doc->>'pointer_document_hash'
                    = p_reconciliation_doc->>'artifact_pointer_document_hash'
       )
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_routing_experiments_v2 experiment
           JOIN public.research_lab_attested_execution_receipts_v2 observation_receipt
             ON observation_receipt.receipt_hash
                    = experiment.execution_envelope_doc
                        ->>'model_binding_observation_receipt_hash'
            AND observation_receipt.role = 'gateway_scoring'
            AND observation_receipt.purpose = 'research_lab.routing_model_binding_observation.v2'
            AND observation_receipt.receipt_status = 'succeeded'
            AND observation_receipt.input_root
                    = experiment.execution_envelope_doc #>>
                        '{model_binding_observation,result,request_root}'
            AND observation_receipt.receipt_doc
                    = experiment.execution_envelope_doc #>
                        '{model_binding_observation,receipt}'
           WHERE experiment.experiment_hash = p_experiment_hash
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_evaluation_mismatch' USING ERRCODE = '23503';
    END IF;
    IF pg_catalog.jsonb_array_length(
        coalesce(evaluation.evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
    ) = 0
       OR pg_catalog.jsonb_array_length(
        coalesce(evaluation.evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
    ) = 0
       OR NOT EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(evaluation.evaluation_doc->'variants') AS variant(value)
        WHERE variant.value->>'variant_id' = p_selected_variant_id
          AND (variant.value->>'passed')::BOOLEAN IS TRUE
          AND coalesce((variant.value #>> '{calibration,adapter_failure_count}')::INTEGER, -1) = 0
          AND coalesce((variant.value #>> '{holdout,adapter_failure_count}')::INTEGER, -1) = 0
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(evaluation.evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
        ) AS ref(receipt_id)
        WHERE NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_decision_receipts_v2 decision
             WHERE decision.experiment_hash = p_experiment_hash
               AND decision.receipt_id = ref.receipt_id
        )
    ) OR EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements_text(
            coalesce(evaluation.evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
        ) AS ref(provider_receipt_ref)
        LEFT JOIN public.research_lab_routing_provider_attempts_v2 attempt
          ON attempt.experiment_hash = p_experiment_hash
         AND attempt.provider_receipt_ref = ref.provider_receipt_ref
        WHERE attempt.provider_receipt_ref IS NULL
           OR attempt.outcome = 'adapter_failure'
    ) OR EXISTS (
        (SELECT decision.receipt_id
           FROM public.research_lab_routing_decision_receipts_v2 decision
          WHERE decision.experiment_hash = p_experiment_hash)
        EXCEPT
        (SELECT ref.receipt_id
           FROM pg_catalog.jsonb_array_elements_text(
               coalesce(evaluation.evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
           ) AS ref(receipt_id))
    ) OR EXISTS (
        (SELECT attempt.provider_receipt_ref
           FROM public.research_lab_routing_provider_attempts_v2 attempt
          WHERE attempt.experiment_hash = p_experiment_hash)
        EXCEPT
        (SELECT ref.provider_receipt_ref
           FROM pg_catalog.jsonb_array_elements_text(
               coalesce(evaluation.evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
           ) AS ref(provider_receipt_ref))
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_routing_provider_attempts_v2 attempt
        WHERE attempt.experiment_hash = p_experiment_hash
          AND (
              attempt.outcome = 'adapter_failure'
              OR attempt.billing_state <> 'known'
          )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_routing_provider_attempts_v2 attempt
        LEFT JOIN public.research_lab_attested_execution_receipts_v2 authorization_receipt
          ON authorization_receipt.receipt_hash = attempt.authorization_proof_hash
         AND authorization_receipt.role = 'gateway_scoring'
         AND authorization_receipt.purpose = 'research_lab.routing_provider_evidence.v2'
         AND authorization_receipt.receipt_status = 'succeeded'
         AND authorization_receipt.input_root = attempt.authorization_hash
         AND authorization_receipt.output_root
                = attempt.attempt_doc #>> '{call_grant_result,output_root}'
         AND authorization_receipt.receipt_doc
                = attempt.attempt_doc->'call_grant_receipt'
        WHERE attempt.experiment_hash = p_experiment_hash
          AND authorization_receipt.receipt_hash IS NULL
    ) OR EXISTS (
        -- Every provider receipt must have one exact reservation identity and
        -- that reservation must terminate in `settle`.  A free-floating
        -- attempt, or an attempt whose budget chain is still open/uncertain,
        -- is never promotion evidence.
        SELECT 1
        FROM public.research_lab_routing_provider_attempts_v2 attempt
        LEFT JOIN public.research_lab_routing_budget_events_v2 reserve_event
          ON reserve_event.experiment_hash = attempt.experiment_hash
         AND reserve_event.reservation_id = attempt.reservation_id
         AND reserve_event.event_type = 'reserve'
        LEFT JOIN LATERAL (
            SELECT head.event_type, head.event_key
            FROM public.research_lab_routing_budget_events_v2 head
            WHERE head.reservation_id = attempt.reservation_id
            ORDER BY head.created_at DESC, head.event_key DESC
            LIMIT 1
        ) budget_head ON TRUE
        WHERE attempt.experiment_hash = p_experiment_hash
          AND (
              reserve_event.event_key IS NULL
              OR reserve_event.binding_id IS DISTINCT FROM attempt.binding_id
              OR reserve_event.claim_key IS DISTINCT FROM attempt.claim_key
              OR reserve_event.claim_generation IS DISTINCT FROM attempt.claim_generation
              OR reserve_event.event_doc->>'unit_ref' IS DISTINCT FROM attempt.unit_ref
              OR reserve_event.event_doc->>'variant_id' IS DISTINCT FROM attempt.variant_id
              OR reserve_event.event_doc->>'request_fingerprint' IS DISTINCT FROM attempt.request_fingerprint
              OR reserve_event.event_doc->>'action_id' IS DISTINCT FROM attempt.action_id
              OR budget_head.event_type IS DISTINCT FROM 'settle'
          )
    ) OR EXISTS (
        -- A receipt chain cannot promote while any budget reservation has an
        -- unresolved head.  In particular, crash recovery retains the full
        -- reservation as `uncertain`; it never releases an unknown provider
        -- debit merely because the original lease expired.
        SELECT 1
        FROM (
            SELECT DISTINCT ON (budget_event.reservation_id)
                budget_event.event_type
            FROM public.research_lab_routing_budget_events_v2 budget_event
            WHERE budget_event.experiment_hash = p_experiment_hash
            ORDER BY budget_event.reservation_id,
                budget_event.created_at DESC, budget_event.event_key DESC
        ) AS budget_head
        WHERE budget_head.event_type <> 'settle'
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_promote_receipt_reconciliation_failed' USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.event_hash = p_event_hash
           AND (
               event.experiment_hash IS DISTINCT FROM p_experiment_hash
               OR event.event_type IS DISTINCT FROM 'promoted'
               OR event.event_doc IS DISTINCT FROM p_event_doc
           )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_promote_event_conflict' USING ERRCODE = '23505';
    END IF;
    SELECT * INTO existing FROM public.research_lab_routing_lab_references_v2
     WHERE reference_hash = p_reference_hash;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.evaluation_receipt_id IS DISTINCT FROM p_evaluation_receipt_id
           OR existing.selected_variant_id IS DISTINCT FROM p_selected_variant_id
           OR existing.reconciliation_doc IS DISTINCT FROM p_reconciliation_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_promote_conflict' USING ERRCODE = '23505';
        END IF;
        IF NOT EXISTS (
            SELECT 1 FROM public.research_lab_routing_experiment_events_v2 event
             WHERE event.event_hash = p_event_hash
               AND event.experiment_hash = p_experiment_hash
               AND event.event_type = 'promoted'
               AND event.event_doc = p_event_doc
        ) THEN
            RAISE EXCEPTION 'research_lab_routing_promote_event_missing' USING ERRCODE = '23503';
        END IF;
        RETURN pg_catalog.jsonb_build_object('reference_hash', p_reference_hash, 'idempotent', TRUE);
    END IF;
    INSERT INTO public.research_lab_routing_lab_references_v2 (
        reference_hash, experiment_hash, evaluation_receipt_id,
        selected_variant_id, reconciliation_doc
    ) VALUES (
        p_reference_hash, p_experiment_hash, p_evaluation_receipt_id,
        p_selected_variant_id, p_reconciliation_doc
    );
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, event_doc
    ) VALUES (p_event_hash, p_experiment_hash, 'promoted', p_event_doc)
    ON CONFLICT (event_hash) DO NOTHING;
    IF NOT EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.event_hash = p_event_hash
           AND event.experiment_hash = p_experiment_hash
           AND event.event_type = 'promoted'
           AND event.event_doc = p_event_doc
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_promote_event_conflict' USING ERRCODE = '23505';
    END IF;
    RETURN pg_catalog.jsonb_build_object('reference_hash', p_reference_hash, 'idempotent', FALSE);
END;
$promote$;

-- V3 claim authority.  The v2 bearer contract remains installed only for
-- non-destructive migration compatibility and is revoked below.  V3 stores
-- only the claim identity and the queue lease identity; service_role is the
-- authentication boundary and the database lease is the fence.
CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_claims_v3 (
    claim_key TEXT PRIMARY KEY CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    request_hash TEXT NOT NULL
        REFERENCES public.research_lab_routing_execution_requests_v2(request_hash),
    lease_hash TEXT NOT NULL CHECK (lease_hash ~ '^sha256:[0-9a-f]{64}$'),
    lease_generation BIGINT NOT NULL CHECK (lease_generation > 0),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    worker_ref TEXT NOT NULL CHECK (worker_ref ~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'),
    claim_state TEXT NOT NULL CHECK (claim_state IN ('claimed', 'recovered')),
    lease_expires_at TIMESTAMPTZ,
    claim_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(claim_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
    CHECK ((claim_state = 'claimed' AND lease_expires_at IS NOT NULL)
        OR (claim_state = 'recovered'))
);
CREATE UNIQUE INDEX IF NOT EXISTS rl_route_claim_v3_generation_uq
    ON public.research_lab_routing_experiment_claims_v3(experiment_hash, claim_generation);
CREATE INDEX IF NOT EXISTS rl_route_claim_v3_head_idx
    ON public.research_lab_routing_experiment_claims_v3(experiment_hash, created_at DESC);

CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_claim_heartbeats_v3 (
    heartbeat_key TEXT PRIMARY KEY CHECK (heartbeat_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    heartbeat_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(heartbeat_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
CREATE INDEX IF NOT EXISTS rl_route_claim_heartbeat_v3_head_idx
    ON public.research_lab_routing_experiment_claim_heartbeats_v3(
        experiment_hash, claim_key, claim_generation, created_at DESC
    );

CREATE TABLE IF NOT EXISTS public.research_lab_routing_experiment_claim_closures_v3 (
    close_key TEXT PRIMARY KEY CHECK (close_key ~ '^sha256:[0-9a-f]{64}$'),
    experiment_hash TEXT NOT NULL
        REFERENCES public.research_lab_routing_experiments_v2(experiment_hash),
    claim_key TEXT NOT NULL CHECK (claim_key ~ '^sha256:[0-9a-f]{64}$'),
    claim_generation BIGINT NOT NULL CHECK (claim_generation > 0),
    close_reason TEXT NOT NULL CHECK (close_reason IN ('completed', 'failed', 'cancelled')),
    close_doc JSONB NOT NULL CHECK (pg_catalog.jsonb_typeof(close_doc) = 'object'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
CREATE UNIQUE INDEX IF NOT EXISTS rl_route_claim_close_v3_once_uq
    ON public.research_lab_routing_experiment_claim_closures_v3(
        experiment_hash, claim_key, claim_generation
    );

CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_claim_v3(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $assert_claim_v3$
DECLARE
    head public.research_lab_routing_experiment_claims_v3%ROWTYPE;
    queue_lease RECORD;
    effective_expiry TIMESTAMPTZ;
    checked_at TIMESTAMPTZ;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_fence_invalid' USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    SELECT * INTO head
      FROM public.research_lab_routing_experiment_claims_v3
     WHERE experiment_hash = p_experiment_hash
     ORDER BY claim_generation DESC, created_at DESC, claim_key DESC
     LIMIT 1;
    IF NOT FOUND
       OR head.claim_state IS DISTINCT FROM 'claimed'
       OR head.claim_key IS DISTINCT FROM p_claim_key
       OR head.claim_generation IS DISTINCT FROM p_claim_generation
       OR EXISTS (
           SELECT 1
             FROM public.research_lab_routing_experiment_claim_closures_v3 closure
            WHERE closure.experiment_hash = p_experiment_hash
              AND closure.claim_key = p_claim_key
              AND closure.claim_generation = p_claim_generation
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_fence_stale' USING ERRCODE = '42501';
    END IF;
    SELECT greatest(
        head.lease_expires_at,
        coalesce(max(heartbeat.lease_expires_at), head.lease_expires_at)
    ) INTO effective_expiry
      FROM public.research_lab_routing_experiment_claim_heartbeats_v3 heartbeat
     WHERE heartbeat.experiment_hash = p_experiment_hash
       AND heartbeat.claim_key = p_claim_key
       AND heartbeat.claim_generation = p_claim_generation;
    -- Migration 159 owns the mutable queue table.  Refer to it dynamically so
    -- migration 157 remains replayable by itself, but fail closed until the
    -- queue authority is installed.  The product claim is usable only while
    -- both database lease clocks and every copied binding field agree.
    IF pg_catalog.to_regclass(
        'public.research_lab_routing_execution_request_leases_v2'
    ) IS NULL THEN
        RAISE EXCEPTION 'research_lab_routing_claim_queue_authority_missing'
            USING ERRCODE = '42501';
    END IF;
    EXECUTE $queue_lease$
        SELECT request_hash, experiment_hash, lease_hash, lease_generation,
               worker_ref, lease_state, lease_expires_at,
               execution_claim_key, execution_claim_generation
          FROM public.research_lab_routing_execution_request_leases_v2
         WHERE request_hash = $1
    $queue_lease$
    INTO queue_lease
    USING head.request_hash;
    checked_at := pg_catalog.clock_timestamp();
    IF queue_lease.request_hash IS NULL
       OR queue_lease.experiment_hash IS DISTINCT FROM head.experiment_hash
       OR queue_lease.lease_hash IS DISTINCT FROM head.lease_hash
       OR queue_lease.lease_generation IS DISTINCT FROM head.lease_generation
       OR queue_lease.worker_ref IS DISTINCT FROM head.worker_ref
       OR queue_lease.execution_claim_key IS DISTINCT FROM head.claim_key
       OR queue_lease.execution_claim_generation IS DISTINCT FROM head.claim_generation
       OR queue_lease.lease_state IS DISTINCT FROM 'claimed'
       OR queue_lease.lease_expires_at IS NULL
       OR queue_lease.lease_expires_at <= checked_at
       OR effective_expiry IS NULL
       OR effective_expiry <= checked_at
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_fence_stale' USING ERRCODE = '42501';
    END IF;
END;
$assert_claim_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_claim_experiment_v3(
    p_experiment_hash TEXT,
    p_request_hash TEXT,
    p_lease_hash TEXT,
    p_lease_generation BIGINT,
    p_claim_key TEXT,
    p_worker_ref TEXT,
    p_lease_seconds INTEGER,
    p_claim_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $claim_v3$
DECLARE
    existing public.research_lab_routing_experiment_claims_v3%ROWTYPE;
    head public.research_lab_routing_experiment_claims_v3%ROWTYPE;
    expiry TIMESTAMPTZ;
    effective_expiry TIMESTAMPTZ;
    next_generation BIGINT;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_lease_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_lease_generation IS NULL OR p_lease_generation < 1
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref IS NULL
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_lease_seconds IS NULL OR p_lease_seconds < 1 OR p_lease_seconds > 3600
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_claim_doc, 'routing claim v3');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing claim event v3');
    IF p_claim_doc->>'request_hash' IS DISTINCT FROM p_request_hash
       OR p_claim_doc->>'lease_hash' IS DISTINCT FROM p_lease_hash
       OR p_claim_doc->>'lease_generation' IS DISTINCT FROM p_lease_generation::TEXT
       OR p_claim_doc->>'worker_ref' IS DISTINCT FROM p_worker_ref
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_document_mismatch'
            USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_execution_requests_v2 request
         WHERE request.request_hash = p_request_hash
           AND request.experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_request_missing' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claims_v3
     WHERE claim_key = p_claim_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.request_hash IS DISTINCT FROM p_request_hash
           OR existing.lease_hash IS DISTINCT FROM p_lease_hash
           OR existing.lease_generation IS DISTINCT FROM p_lease_generation
           OR existing.worker_ref IS DISTINCT FROM p_worker_ref
           OR existing.claim_doc IS DISTINCT FROM p_claim_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_conflict' USING ERRCODE = '23505';
        END IF;
        IF existing.claim_state = 'recovered' THEN
            RETURN pg_catalog.jsonb_build_object(
                'claimed', FALSE, 'idempotent', TRUE, 'terminal', TRUE,
                'claim_key', existing.claim_key,
                'claim_generation', existing.claim_generation,
                'request_hash', existing.request_hash,
                'lease_hash', existing.lease_hash,
                'lease_generation', existing.lease_generation
            );
        END IF;
        SELECT greatest(
            existing.lease_expires_at,
            coalesce(max(heartbeat.lease_expires_at), existing.lease_expires_at)
        ) INTO effective_expiry
          FROM public.research_lab_routing_experiment_claim_heartbeats_v3 heartbeat
         WHERE heartbeat.experiment_hash = existing.experiment_hash
           AND heartbeat.claim_key = existing.claim_key
           AND heartbeat.claim_generation = existing.claim_generation;
        IF effective_expiry <= pg_catalog.clock_timestamp() THEN
            RETURN pg_catalog.jsonb_build_object(
                'claimed', FALSE, 'idempotent', TRUE, 'recoverable', TRUE,
                'claim_key', existing.claim_key,
                'claim_generation', existing.claim_generation,
                'request_hash', existing.request_hash,
                'lease_hash', existing.lease_hash,
                'lease_generation', existing.lease_generation,
                'lease_expires_at', effective_expiry
            );
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'claimed', TRUE,
            'idempotent', TRUE,
            'terminal', existing.claim_state = 'recovered',
            'claim_key', existing.claim_key,
            'claim_generation', existing.claim_generation,
            'request_hash', existing.request_hash,
            'lease_hash', existing.lease_hash,
            'lease_generation', existing.lease_generation,
            'lease_expires_at', effective_expiry
        );
    END IF;
    SELECT * INTO head
      FROM public.research_lab_routing_experiment_claims_v3
     WHERE experiment_hash = p_experiment_hash
     ORDER BY claim_generation DESC, created_at DESC, claim_key DESC
     LIMIT 1;
    IF FOUND THEN
        SELECT greatest(
            head.lease_expires_at,
            coalesce(max(heartbeat.lease_expires_at), head.lease_expires_at)
        ) INTO effective_expiry
          FROM public.research_lab_routing_experiment_claim_heartbeats_v3 heartbeat
         WHERE heartbeat.experiment_hash = head.experiment_hash
           AND heartbeat.claim_key = head.claim_key
           AND heartbeat.claim_generation = head.claim_generation;
        IF head.claim_state = 'recovered' THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_recovered' USING ERRCODE = '23505';
        ELSIF effective_expiry > pg_catalog.clock_timestamp() THEN
            RETURN pg_catalog.jsonb_build_object(
                'claimed', FALSE, 'idempotent', FALSE,
                'claim_key', head.claim_key,
                'claim_generation', head.claim_generation,
                'request_hash', head.request_hash,
                'lease_hash', head.lease_hash,
                'lease_generation', head.lease_generation,
                'lease_expires_at', effective_expiry
            );
        ELSE
            RAISE EXCEPTION 'research_lab_routing_claim_v3_expired_requires_recovery'
                USING ERRCODE = '23505';
        END IF;
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_claim_closures_v3 closure
         WHERE closure.experiment_hash = p_experiment_hash
    ) OR EXISTS (
        SELECT 1 FROM public.research_lab_routing_evaluation_receipts_v2 evaluation
         WHERE evaluation.experiment_hash = p_experiment_hash
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_terminal' USING ERRCODE = '23505';
    END IF;
    SELECT coalesce(max(claim_generation), 0) + 1 INTO next_generation
      FROM public.research_lab_routing_experiment_claims_v3
     WHERE experiment_hash = p_experiment_hash;
    expiry := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_seconds);
    INSERT INTO public.research_lab_routing_experiment_claims_v3 (
        claim_key, experiment_hash, request_hash, lease_hash, lease_generation,
        claim_generation, worker_ref, claim_state, lease_expires_at, claim_doc
    ) VALUES (
        p_claim_key, p_experiment_hash, p_request_hash, p_lease_hash,
        p_lease_generation, next_generation, p_worker_ref, 'claimed', expiry,
        p_claim_doc
    );
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, claim_key, claim_generation, event_doc
    ) VALUES (p_event_hash, p_experiment_hash, 'claimed', p_claim_key, next_generation, p_event_doc)
    ON CONFLICT (event_hash) DO NOTHING;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.event_hash = p_event_hash
           AND (event.experiment_hash IS DISTINCT FROM p_experiment_hash
             OR event.event_type IS DISTINCT FROM 'claimed'
             OR event.claim_key IS DISTINCT FROM p_claim_key
             OR event.claim_generation IS DISTINCT FROM next_generation
             OR event.event_doc IS DISTINCT FROM p_event_doc)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_event_conflict' USING ERRCODE = '23505';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'claimed', TRUE, 'idempotent', FALSE,
        'claim_key', p_claim_key, 'claim_generation', next_generation,
        'request_hash', p_request_hash, 'lease_hash', p_lease_hash,
        'lease_generation', p_lease_generation, 'lease_expires_at', expiry
    );
END;
$claim_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_recover_claim_v3(
    p_experiment_hash TEXT,
    p_recovery_key TEXT,
    p_worker_ref TEXT,
    p_recovery_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $recover_claim_v3$
DECLARE
    head public.research_lab_routing_experiment_claims_v3%ROWTYPE;
    existing public.research_lab_routing_experiment_claims_v3%ROWTYPE;
    queue_lease RECORD;
    effective_claim_expiry TIMESTAMPTZ;
    checked_at TIMESTAMPTZ;
    queue_update_count INTEGER;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_recovery_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_worker_ref IS NULL
       OR p_worker_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_invalid' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_recovery_doc, 'routing claim recovery v3');
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing claim recovery event v3');
    IF p_recovery_doc->>'schema_version'
            IS DISTINCT FROM 'leadpoet.research_lab.routing_claim_recovery.v3'
       OR p_recovery_doc->>'worker_ref' IS DISTINCT FROM p_worker_ref
       OR p_recovery_doc->>'stale_claim_key' !~ '^sha256:[0-9a-f]{64}$'
       OR p_recovery_key IS NOT DISTINCT FROM p_recovery_doc->>'stale_claim_key'
       OR coalesce((p_recovery_doc->>'stale_claim_generation')::BIGINT, 0) < 1
       OR p_event_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR p_event_doc->>'recovery_key' IS DISTINCT FROM p_recovery_key
       OR p_event_doc->>'worker_ref' IS DISTINCT FROM p_worker_ref
       OR p_event_doc->>'event_type' IS DISTINCT FROM 'claim_recovered'
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_document_mismatch'
            USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claims_v3
     WHERE claim_key = p_recovery_key;
    IF FOUND THEN
        IF existing.claim_state IS DISTINCT FROM 'recovered'
           OR existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.worker_ref IS DISTINCT FROM p_worker_ref
           OR existing.claim_doc IS DISTINCT FROM p_recovery_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_conflict'
                USING ERRCODE = '23505';
        END IF;
        IF NOT EXISTS (
            SELECT 1
              FROM public.research_lab_routing_experiment_events_v2 event
             WHERE event.event_hash = p_event_hash
               AND event.experiment_hash = p_experiment_hash
               AND event.event_type = 'claim_recovered'
               AND event.claim_key = p_recovery_key
               AND event.claim_generation = existing.claim_generation
               AND event.event_doc = p_event_doc
        ) OR EXISTS (
            SELECT 1
              FROM (
                  SELECT DISTINCT ON (budget_event.reservation_id)
                      budget_event.event_type
                    FROM public.research_lab_routing_budget_events_v2 budget_event
                   WHERE budget_event.experiment_hash = p_experiment_hash
                   ORDER BY budget_event.reservation_id,
                       budget_event.created_at DESC, budget_event.event_key DESC
              ) budget_head
             WHERE budget_head.event_type = 'reserve'
        ) THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_replay_incomplete'
                USING ERRCODE = '23503';
        END IF;
        IF pg_catalog.to_regclass(
            'public.research_lab_routing_execution_request_leases_v2'
        ) IS NULL THEN
            RAISE EXCEPTION 'research_lab_routing_claim_queue_authority_missing'
                USING ERRCODE = '42501';
        END IF;
        EXECUTE $queue_replay$
            SELECT request_hash, experiment_hash, lease_hash, lease_generation,
                   worker_ref, lease_state, close_reason,
                   execution_claim_key, execution_claim_generation
              FROM public.research_lab_routing_execution_request_leases_v2
             WHERE request_hash = $1
        $queue_replay$
        INTO queue_lease
        USING existing.request_hash;
        IF queue_lease.request_hash IS NULL
           OR queue_lease.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR queue_lease.lease_hash IS DISTINCT FROM existing.lease_hash
           OR queue_lease.lease_generation IS DISTINCT FROM existing.lease_generation
           OR queue_lease.lease_state IS DISTINCT FROM 'recovered'
           OR queue_lease.close_reason IS DISTINCT FROM 'recovered'
           OR queue_lease.execution_claim_key IS DISTINCT FROM
                p_recovery_doc->>'stale_claim_key'
           OR queue_lease.execution_claim_generation IS DISTINCT FROM
                (p_recovery_doc->>'stale_claim_generation')::BIGINT
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_queue_mismatch'
                USING ERRCODE = '23503';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'recovered', TRUE, 'idempotent', TRUE,
            'terminal', TRUE, 'billing_state', 'uncertain',
            'recovery_key', p_recovery_key,
            'claim_generation', existing.claim_generation
        );
    END IF;
    SELECT * INTO head
      FROM public.research_lab_routing_experiment_claims_v3
     WHERE experiment_hash = p_experiment_hash
     ORDER BY claim_generation DESC, created_at DESC, claim_key DESC
     LIMIT 1;
    IF NOT FOUND OR head.claim_state IS DISTINCT FROM 'claimed' THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_not_expired'
            USING ERRCODE = '23505';
    END IF;
    IF p_recovery_doc->>'stale_claim_key' IS DISTINCT FROM head.claim_key
       OR (p_recovery_doc->>'stale_claim_generation')::BIGINT
            IS DISTINCT FROM head.claim_generation
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_head_mismatch'
            USING ERRCODE = '23505';
    END IF;
    SELECT greatest(
        head.lease_expires_at,
        coalesce(max(heartbeat.lease_expires_at), head.lease_expires_at)
    ) INTO effective_claim_expiry
      FROM public.research_lab_routing_experiment_claim_heartbeats_v3 heartbeat
     WHERE heartbeat.experiment_hash = p_experiment_hash
       AND heartbeat.claim_key = head.claim_key
       AND heartbeat.claim_generation = head.claim_generation;
    IF pg_catalog.to_regclass(
        'public.research_lab_routing_execution_request_leases_v2'
    ) IS NULL THEN
        RAISE EXCEPTION 'research_lab_routing_claim_queue_authority_missing'
            USING ERRCODE = '42501';
    END IF;
    EXECUTE $queue_recovery$
        SELECT request_hash, experiment_hash, lease_hash, lease_generation,
               worker_ref, lease_state, lease_expires_at, close_reason,
               execution_claim_key, execution_claim_generation
          FROM public.research_lab_routing_execution_request_leases_v2
         WHERE request_hash = $1
         FOR UPDATE
    $queue_recovery$
    INTO queue_lease
    USING head.request_hash;
    checked_at := pg_catalog.clock_timestamp();
    IF queue_lease.request_hash IS NULL
       OR queue_lease.experiment_hash IS DISTINCT FROM head.experiment_hash
       OR queue_lease.lease_hash IS DISTINCT FROM head.lease_hash
       OR queue_lease.lease_generation IS DISTINCT FROM head.lease_generation
       OR queue_lease.worker_ref IS DISTINCT FROM head.worker_ref
       OR queue_lease.execution_claim_key IS DISTINCT FROM head.claim_key
       OR queue_lease.execution_claim_generation IS DISTINCT FROM head.claim_generation
       OR queue_lease.lease_state IS DISTINCT FROM 'claimed'
       OR (queue_lease.lease_expires_at > checked_at
           AND effective_claim_expiry > checked_at)
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_not_expired'
            USING ERRCODE = '23505';
    END IF;
    -- Recovery is terminal.  Every still-open reservation becomes a full
    -- ceiling uncertain charge before the queue lease is closed.  No amount
    -- is released and this experiment can never be leased again.
    INSERT INTO public.research_lab_routing_experiment_claims_v3 (
        claim_key, experiment_hash, request_hash, lease_hash, lease_generation,
        claim_generation, worker_ref, claim_state, lease_expires_at, claim_doc
    ) VALUES (
        p_recovery_key, p_experiment_hash, head.request_hash, head.lease_hash,
        head.lease_generation, head.claim_generation + 1, p_worker_ref,
        'recovered', NULL, p_recovery_doc
    );
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type, credit_microunits, event_doc
    )
    SELECT
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.research_lab.routing_budget_recovery.v3',
                'event_type', 'uncertain',
                'reservation_id', budget_head.reservation_id,
                'recovery_key', p_recovery_key,
                'reserve_event_key', budget_head.event_key
            )
        ),
        budget_head.reservation_id, p_experiment_hash, budget_head.binding_id,
        p_recovery_key, head.claim_generation + 1, 'uncertain',
        budget_head.credit_microunits,
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_budget_event.v3',
            'billing_state', 'uncertain',
            'reason_code', 'claim_recovered_unknown_billing',
            'reservation_id', budget_head.reservation_id,
            'binding_id', budget_head.binding_id,
            'recovery_key', p_recovery_key,
            'reserve_event_key', budget_head.event_key,
            'credit_microunits', budget_head.credit_microunits
        )
      FROM (
          SELECT DISTINCT ON (budget_event.reservation_id) budget_event.*
            FROM public.research_lab_routing_budget_events_v2 budget_event
           WHERE budget_event.experiment_hash = p_experiment_hash
           ORDER BY budget_event.reservation_id,
               budget_event.created_at DESC, budget_event.event_key DESC
      ) budget_head
     WHERE budget_head.event_type = 'reserve';
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, claim_key, claim_generation, event_doc
    ) VALUES (
        p_event_hash, p_experiment_hash, 'claim_recovered', p_recovery_key,
        head.claim_generation + 1, p_event_doc
    ) ON CONFLICT (event_hash) DO NOTHING;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.event_hash = p_event_hash
           AND (event.experiment_hash IS DISTINCT FROM p_experiment_hash
             OR event.event_type IS DISTINCT FROM 'claim_recovered'
             OR event.claim_key IS DISTINCT FROM p_recovery_key
             OR event.claim_generation IS DISTINCT FROM head.claim_generation + 1
             OR event.event_doc IS DISTINCT FROM p_event_doc)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_event_conflict'
            USING ERRCODE = '23505';
    END IF;
    EXECUTE $close_queue$
        UPDATE public.research_lab_routing_execution_request_leases_v2
           SET lease_state = 'recovered',
               lease_expires_at = NULL,
               close_reason = 'recovered',
               updated_at = pg_catalog.clock_timestamp()
         WHERE request_hash = $1
           AND experiment_hash = $2
           AND lease_hash = $3
           AND lease_generation = $4
           AND worker_ref = $5
           AND lease_state = 'claimed'
           AND execution_claim_key = $6
           AND execution_claim_generation = $7
    $close_queue$
    USING head.request_hash, head.experiment_hash, head.lease_hash,
          head.lease_generation, head.worker_ref, head.claim_key,
          head.claim_generation;
    GET DIAGNOSTICS queue_update_count = ROW_COUNT;
    IF queue_update_count <> 1 THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_recovery_queue_close_failed'
            USING ERRCODE = '40001';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'recovered', TRUE, 'idempotent', FALSE,
        'terminal', TRUE,
        'recovery_key', p_recovery_key,
        'claim_generation', head.claim_generation + 1,
        'billing_state', 'uncertain'
    );
END;
$recover_claim_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_renew_claim_v3(
    p_heartbeat_key TEXT,
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_lease_seconds INTEGER,
    p_heartbeat_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $renew_claim_v3$
DECLARE
    existing public.research_lab_routing_experiment_claim_heartbeats_v3%ROWTYPE;
    expiry TIMESTAMPTZ;
BEGIN
    IF p_heartbeat_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR p_lease_seconds IS NULL OR p_lease_seconds < 1 OR p_lease_seconds > 3600
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_renew_invalid' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_heartbeat_doc, 'routing claim heartbeat v3');
    expiry := pg_catalog.clock_timestamp() + pg_catalog.make_interval(secs => p_lease_seconds);
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claim_heartbeats_v3
     WHERE heartbeat_key = p_heartbeat_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.heartbeat_doc IS DISTINCT FROM p_heartbeat_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_renew_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'renewed', TRUE, 'idempotent', TRUE,
            'heartbeat_key', p_heartbeat_key, 'lease_expires_at', existing.lease_expires_at
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    INSERT INTO public.research_lab_routing_experiment_claim_heartbeats_v3 (
        heartbeat_key, experiment_hash, claim_key, claim_generation,
        lease_expires_at, heartbeat_doc
    ) VALUES (
        p_heartbeat_key, p_experiment_hash, p_claim_key, p_claim_generation,
        expiry, p_heartbeat_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'renewed', TRUE, 'idempotent', FALSE,
        'heartbeat_key', p_heartbeat_key, 'lease_expires_at', expiry
    );
END;
$renew_claim_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_close_claim_v3(
    p_close_key TEXT,
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_close_reason TEXT,
    p_close_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $close_claim_v3$
DECLARE
    existing public.research_lab_routing_experiment_claim_closures_v3%ROWTYPE;
BEGIN
    IF p_close_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR p_close_reason NOT IN ('completed', 'failed', 'cancelled')
    THEN
        RAISE EXCEPTION 'research_lab_routing_claim_v3_close_invalid' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_close_doc, 'routing claim close v3');
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_claim_closures_v3
     WHERE close_key = p_close_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.close_reason IS DISTINCT FROM p_close_reason
           OR existing.close_doc IS DISTINCT FROM p_close_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_claim_v3_close_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('closed', TRUE, 'idempotent', TRUE, 'close_key', p_close_key);
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    INSERT INTO public.research_lab_routing_experiment_claim_closures_v3 (
        close_key, experiment_hash, claim_key, claim_generation, close_reason, close_doc
    ) VALUES (
        p_close_key, p_experiment_hash, p_claim_key, p_claim_generation,
        p_close_reason, p_close_doc
    );
    RETURN pg_catalog.jsonb_build_object('closed', TRUE, 'idempotent', FALSE, 'close_key', p_close_key);
END;
$close_claim_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_fenced_event_v3(
    p_event_hash TEXT,
    p_experiment_hash TEXT,
    p_event_type TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $append_fenced_event_v3$
DECLARE
    existing public.research_lab_routing_experiment_events_v2%ROWTYPE;
BEGIN
    IF p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_type NOT IN (
           'run_started', 'run_completed', 'run_failed', 'promotion_requested',
           'provider_dispatch_started'
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_event_v3_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_event_doc, 'routing fenced event v3');
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_experiment_events_v2
     WHERE event_hash = p_event_hash;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.event_type IS DISTINCT FROM p_event_type
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_event_v3_conflict' USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object('event_hash', p_event_hash, 'idempotent', TRUE);
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    IF p_event_type = 'provider_dispatch_started' THEN
        IF p_event_doc->>'reservation_id'
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'binding_id'
                !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
           OR p_event_doc->>'request_fingerprint'
                !~ '^sha256:[0-9a-f]{64}$'
           OR NOT EXISTS (
               SELECT 1
                 FROM public.research_lab_routing_budget_events_v2 reserve_event
                WHERE reserve_event.experiment_hash = p_experiment_hash
                  AND reserve_event.reservation_id = p_event_doc->>'reservation_id'
                  AND reserve_event.binding_id = p_event_doc->>'binding_id'
                  AND reserve_event.claim_key = p_claim_key
                  AND reserve_event.claim_generation = p_claim_generation
                  AND reserve_event.event_type = 'reserve'
                  AND reserve_event.event_doc = p_event_doc
           )
        THEN
            RAISE EXCEPTION 'research_lab_routing_dispatch_event_v3_reservation_mismatch'
                USING ERRCODE = '23503';
        END IF;
    END IF;
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, claim_key, claim_generation, event_doc
    ) VALUES (
        p_event_hash, p_experiment_hash, p_event_type, p_claim_key, p_claim_generation, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object('event_hash', p_event_hash, 'idempotent', FALSE);
END;
$append_fenced_event_v3$;

-- V3 provider receipt graph.  This validator intentionally has no legacy
-- terminal_proof or call_grant.job_id path.  The admission, authorization,
-- and terminal jobs are separate signed executions with exact ancestry.
CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v3(
    p_experiment_hash TEXT,
    p_binding_id TEXT,
    p_tool_id TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_action_id TEXT,
    p_authorization_hash TEXT,
    p_authorization_request_hash TEXT,
    p_authorization_proof_hash TEXT,
    p_terminal_receipt_hash TEXT,
    p_protected_release_receipt_hash TEXT,
    p_admission_bundle_hash TEXT,
    p_terminal_request_hash TEXT,
    p_terminal_result_hash TEXT,
    p_terminal_provider_record_hash TEXT,
    p_terminal_billing_projection_hash TEXT,
    p_outcome TEXT,
    p_credit_microunits BIGINT,
    p_latency_ms BIGINT,
    p_billing_state TEXT,
    p_authoritative_billed_credit_microunits BIGINT,
    p_attempt_doc JSONB
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $receipt_chain_v3$
DECLARE
    authorization_doc JSONB := p_attempt_doc->'call_grant_receipt';
    protected_doc JSONB := p_attempt_doc->'protected_release_receipt';
    terminal_doc JSONB := p_attempt_doc->'terminal_execution_receipt';
    terminal_result JSONB := p_attempt_doc->'terminal_result';
    admission JSONB := p_attempt_doc->'admission_bundle';
    call_grant_doc JSONB := p_attempt_doc->'call_grant';
    grant_result JSONB := p_attempt_doc->'call_grant_result';
    budget_reservation JSONB := p_attempt_doc->'terminal_result'->'budget_reservation';
    experiment_envelope JSONB;
    experiment_envelope_hash TEXT;
    model_observation_hash TEXT;
    admission_job_id TEXT;
    authorization_job_id TEXT;
    terminal_job_id TEXT;
    expected_terminal_job_id TEXT;
    billing_projection JSONB;
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_proof_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_protected_release_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_admission_bundle_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_result_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_provider_record_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_billing_projection_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_attempt_doc->>'schema_version'
            IS DISTINCT FROM 'leadpoet.research_lab.routing_provider_attempt.v3'
       OR pg_catalog.jsonb_typeof(authorization_doc) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(protected_doc) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(terminal_doc) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(terminal_result) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(admission) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(call_grant_doc) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(grant_result) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(budget_reservation) IS DISTINCT FROM 'object'
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_graph_shape_invalid'
            USING ERRCODE = '22023';
    END IF;
    IF p_attempt_doc ? 'terminal_proof'
       OR call_grant_doc ? 'job_id'
       OR p_authorization_hash IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(call_grant_doc)
       OR p_attempt_doc->>'authorization_request_hash'
            IS DISTINCT FROM p_authorization_request_hash
       OR p_attempt_doc->>'terminal_request_hash'
            IS DISTINCT FROM p_terminal_request_hash
       OR p_terminal_request_hash = p_authorization_request_hash
       OR p_terminal_result_hash IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(terminal_result)
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_request_binding_mismatch'
            USING ERRCODE = '22023';
    END IF;
    IF coalesce(call_grant_doc->>'claim_key', '')
            !~ '^sha256:[0-9a-f]{64}$'
       OR coalesce(call_grant_doc->>'claim_generation', '')
            !~ '^[1-9][0-9]*$'
       OR coalesce(call_grant_doc->>'claim_fence_hash', '')
            !~ '^sha256:[0-9a-f]{64}$'
       OR coalesce(grant_result->>'claim_generation', '')
            !~ '^[1-9][0-9]*$'
       OR coalesce(grant_result->>'claim_fence_hash', '')
            !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_claim_fence_shape_invalid'
            USING ERRCODE = '22023';
    END IF;
    IF call_grant_doc->>'claim_fence_hash' IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.research_lab.routing_claim_fence.v3',
                    'experiment_hash', p_experiment_hash,
                    'claim_key', call_grant_doc->>'claim_key',
                    'claim_generation', (call_grant_doc->>'claim_generation')::BIGINT
                )
            )
       OR grant_result->>'claim_fence_hash'
            IS DISTINCT FROM call_grant_doc->>'claim_fence_hash'
       OR (grant_result->>'claim_generation')::BIGINT
            IS DISTINCT FROM (call_grant_doc->>'claim_generation')::BIGINT
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_claim_fence_mismatch'
            USING ERRCODE = '22023';
    END IF;
    SELECT experiment.execution_envelope_doc, experiment.execution_envelope_hash
      INTO experiment_envelope, experiment_envelope_hash
      FROM public.research_lab_routing_experiments_v2 experiment
     WHERE experiment.experiment_hash = p_experiment_hash;
    model_observation_hash := experiment_envelope->>'model_binding_observation_receipt_hash';
    admission_job_id := admission->>'job_id';
    authorization_job_id := grant_result->>'authorization_job_id';
    terminal_job_id := terminal_doc->>'job_id';
    expected_terminal_job_id := 'routing-dispatch:' || pg_catalog.substr(
        public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'schema_version', 'leadpoet.routing_provider_dispatch_job.v3',
                'authorization_hash', p_authorization_hash,
                'authorization_proof_hash', p_authorization_proof_hash,
                'authorization_receipt_hash', p_authorization_proof_hash
            )
        ), 8, 32
    );
    IF model_observation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR coalesce(admission_job_id, '')
            !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR coalesce(authorization_job_id, '')
            !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR coalesce(terminal_job_id, '')
            !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR terminal_job_id IS DISTINCT FROM expected_terminal_job_id
       OR public.research_lab_routing_jsonb_hash_v2(admission)
            IS DISTINCT FROM p_admission_bundle_hash
       OR admission->>'schema_version'
            IS DISTINCT FROM 'leadpoet.research_lab.routing_admission.v2'
       OR admission->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR admission->>'envelope_hash' IS DISTINCT FROM experiment_envelope_hash
       OR admission->>'model_binding_observation_receipt_hash'
            IS DISTINCT FROM model_observation_hash
       OR admission->>'protected_receipt_hash'
            IS DISTINCT FROM p_protected_release_receipt_hash
       OR admission->>'role' IS DISTINCT FROM 'gateway_scoring'
       OR admission->>'purpose'
            IS DISTINCT FROM 'research_lab.routing_provider_evidence.v2'
       OR call_grant_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR call_grant_doc->>'schema_version'
            IS DISTINCT FROM 'leadpoet.routing_provider_call_grant.v2'
       OR call_grant_doc->>'purpose'
            IS DISTINCT FROM 'research_lab.routing_provider_evidence.v2'
       OR call_grant_doc->>'envelope_hash' IS DISTINCT FROM experiment_envelope_hash
       OR call_grant_doc->>'admission_job_id' IS DISTINCT FROM admission_job_id
       OR call_grant_doc->>'admission_bundle_hash' IS DISTINCT FROM p_admission_bundle_hash
       OR authorization_job_id = admission_job_id
       OR terminal_job_id IN (admission_job_id, authorization_job_id)
       OR call_grant_doc->>'protected_release_hash'
            IS DISTINCT FROM admission->>'protected_release_hash'
       OR call_grant_doc->>'model_binding_observation_receipt_hash'
            IS DISTINCT FROM model_observation_hash
       OR call_grant_doc->>'binding_catalog_manifest_hash'
            IS DISTINCT FROM admission->>'binding_catalog_manifest_hash'
       OR call_grant_doc->>'unit_dataset_manifest_hash'
            IS DISTINCT FROM admission->>'unit_dataset_manifest_hash'
       OR call_grant_doc->>'unit_set_hash' IS DISTINCT FROM admission->>'unit_set_hash'
       OR call_grant_doc->'binding'->>'binding_id' IS DISTINCT FROM p_binding_id
       OR call_grant_doc->'binding'->>'tool_id' IS DISTINCT FROM p_tool_id
       OR call_grant_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR call_grant_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR call_grant_doc->>'action_id' IS DISTINCT FROM p_action_id
       OR call_grant_doc->>'request_body_hash'
            IS DISTINCT FROM p_attempt_doc->>'request_body_hash'
       OR grant_result->>'schema_version'
            IS DISTINCT FROM 'leadpoet.routing_provider_call_grant_result.v2'
       OR grant_result->>'operation'
            IS DISTINCT FROM 'attest_routing_provider_call_v2'
       OR grant_result->>'purpose'
            IS DISTINCT FROM 'research_lab.routing_provider_evidence.v2'
       OR grant_result->'attested' IS DISTINCT FROM 'true'::JSONB
       OR grant_result->>'authorization_hash' IS DISTINCT FROM p_authorization_hash
       OR grant_result->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR grant_result->>'admission_job_id' IS DISTINCT FROM admission_job_id
       OR grant_result->>'admission_bundle_hash' IS DISTINCT FROM p_admission_bundle_hash
       OR grant_result->>'protected_release_hash'
            IS DISTINCT FROM admission->>'protected_release_hash'
       OR grant_result->>'model_binding_observation_receipt_hash'
            IS DISTINCT FROM model_observation_hash
       OR grant_result->>'binding_id' IS DISTINCT FROM p_binding_id
       OR grant_result->>'variant_id' IS DISTINCT FROM p_variant_id
       OR grant_result->>'action_id' IS DISTINCT FROM p_action_id
       OR grant_result->>'request_body_hash'
            IS DISTINCT FROM p_attempt_doc->>'request_body_hash'
       OR grant_result->>'output_root' IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(grant_result - 'output_root')
       OR authorization_doc->>'receipt_hash' IS DISTINCT FROM p_authorization_proof_hash
       OR authorization_doc->>'job_id' IS DISTINCT FROM authorization_job_id
       OR authorization_doc->>'input_root' IS DISTINCT FROM p_authorization_request_hash
       OR authorization_doc->>'output_root'
            IS DISTINCT FROM grant_result->>'output_root'
       OR authorization_doc->'parent_receipt_hashes' IS DISTINCT FROM
            pg_catalog.jsonb_build_array(p_protected_release_receipt_hash, model_observation_hash)
       OR terminal_doc->>'receipt_hash' IS DISTINCT FROM p_terminal_receipt_hash
       OR terminal_doc->>'input_root' IS DISTINCT FROM p_terminal_request_hash
       OR terminal_doc->>'output_root' IS DISTINCT FROM p_terminal_result_hash
       OR terminal_doc->'parent_receipt_hashes' IS DISTINCT FROM
            pg_catalog.jsonb_build_array(p_authorization_proof_hash)
       OR protected_doc->>'receipt_hash' IS DISTINCT FROM p_protected_release_receipt_hash
       OR protected_doc->>'job_id' IS DISTINCT FROM admission_job_id
       OR protected_doc->>'role' IS DISTINCT FROM admission->>'role'
       OR protected_doc->>'purpose' IS DISTINCT FROM admission->>'purpose'
       OR protected_doc->>'status' IS DISTINCT FROM 'succeeded'
       OR protected_doc->>'commit_sha' IS DISTINCT FROM admission->>'protected_commit_sha'
       OR protected_doc->>'pcr0' IS DISTINCT FROM admission->>'protected_pcr0'
       OR protected_doc->>'build_manifest_hash'
            IS DISTINCT FROM admission->>'protected_build_manifest_hash'
       OR protected_doc->>'dependency_lock_hash'
            IS DISTINCT FROM admission->>'protected_dependency_lock_hash'
       OR protected_doc->>'config_hash'
            IS DISTINCT FROM admission->>'protected_config_hash'
       OR protected_doc->>'boot_identity_hash'
            IS DISTINCT FROM admission->>'protected_boot_identity_hash'
       OR protected_doc->>'enclave_pubkey'
            IS DISTINCT FROM admission->>'protected_enclave_pubkey'
       OR admission->>'protected_release_hash' IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.routing_protected_release.v2',
                    'protected_receipt_hash', protected_doc->>'receipt_hash',
                    'protected_commit_sha', protected_doc->>'commit_sha',
                    'protected_pcr0', protected_doc->>'pcr0',
                    'protected_build_manifest_hash', protected_doc->>'build_manifest_hash',
                    'protected_dependency_lock_hash', protected_doc->>'dependency_lock_hash',
                    'protected_config_hash', protected_doc->>'config_hash',
                    'protected_boot_identity_hash', protected_doc->>'boot_identity_hash',
                    'protected_enclave_pubkey', protected_doc->>'enclave_pubkey'
                )
            )
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_execution_binding_mismatch'
            USING ERRCODE = '22023';
    END IF;
    billing_projection := terminal_result->'projection';
    IF pg_catalog.jsonb_typeof(billing_projection) IS DISTINCT FROM 'object'
       OR coalesce(billing_projection->>'credit_microunits', '')
            !~ '^[0-9]+$'
       OR coalesce(billing_projection->>'latency_ms', '')
            !~ '^[0-9]+$'
       OR coalesce(
            terminal_result->'provider_receipt'->>'credit_microunits', ''
          ) !~ '^[0-9]+$'
       OR coalesce(
            terminal_result->'provider_receipt'->>'latency_ms', ''
          ) !~ '^[0-9]+$'
       OR coalesce(terminal_result->>'transport_attempt_hash', '')
            !~ '^sha256:[0-9a-f]{64}$'
       OR coalesce(budget_reservation->>'claim_generation', '')
            !~ '^[1-9][0-9]*$'
       OR coalesce(budget_reservation->>'credit_microunits', '')
            !~ '^[0-9]+$'
       OR coalesce(budget_reservation->>'response_hash', '')
            !~ '^sha256:[0-9a-f]{64}$'
       OR coalesce(budget_reservation->>'transport_attempt_hash', '')
            !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_billing_shape_invalid'
            USING ERRCODE = '22023';
    END IF;
    IF terminal_result->>'schema_version'
            IS DISTINCT FROM 'leadpoet.routing_provider_terminal_result.v2'
       OR terminal_result->>'operation'
            IS DISTINCT FROM 'routing_provider_terminal_v2'
       OR coalesce(terminal_result->>'terminal_status', '')
            NOT IN ('authenticated_response', 'transport_failure')
       OR terminal_result->>'authorization_hash' IS DISTINCT FROM p_authorization_hash
       OR terminal_result->>'authorization_proof_hash' IS DISTINCT FROM p_authorization_proof_hash
       OR terminal_result->>'provider_record_hash'
            IS DISTINCT FROM p_terminal_provider_record_hash
       OR p_terminal_billing_projection_hash
            IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(billing_projection)
       OR terminal_result->'binding'->>'binding_id' IS DISTINCT FROM p_binding_id
       OR terminal_result->'binding'->>'tool_id' IS DISTINCT FROM p_tool_id
       OR terminal_result->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR terminal_result->>'request_fingerprint'
            IS DISTINCT FROM p_attempt_doc->>'request_fingerprint'
       OR terminal_result->'provider_receipt'
            IS DISTINCT FROM p_attempt_doc->'provider_receipt'
       OR billing_projection->>'outcome' IS DISTINCT FROM p_outcome
       OR (billing_projection->>'credit_microunits')::BIGINT IS DISTINCT FROM p_credit_microunits
       OR (billing_projection->>'latency_ms')::BIGINT IS DISTINCT FROM p_latency_ms
       OR billing_projection->>'billing_state' IS DISTINCT FROM 'known'
       OR billing_projection->>'binding_id' IS DISTINCT FROM p_binding_id
       OR billing_projection->>'tool_id' IS DISTINCT FROM p_tool_id
       OR billing_projection->>'request_fingerprint'
            IS DISTINCT FROM p_attempt_doc->>'request_fingerprint'
       OR terminal_result->'provider_receipt'->>'binding_id' IS DISTINCT FROM p_binding_id
       OR terminal_result->'provider_receipt'->>'tool_id' IS DISTINCT FROM p_tool_id
       OR terminal_result->'provider_receipt'->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR terminal_result->'provider_receipt'->>'request_fingerprint'
            IS DISTINCT FROM p_attempt_doc->>'request_fingerprint'
       OR terminal_result->'provider_receipt'->>'outcome' IS DISTINCT FROM p_outcome
       OR terminal_result->'provider_receipt'->>'evidence_hash'
            IS DISTINCT FROM billing_projection->>'evidence_hash'
       OR budget_reservation->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.routing_budget_reservation_proof.v3'
       OR (
            SELECT pg_catalog.count(*)
              FROM pg_catalog.jsonb_object_keys(budget_reservation)
          ) <> 11
       OR (terminal_result->'provider_receipt'->>'credit_microunits')::BIGINT
            IS DISTINCT FROM p_credit_microunits
       OR (terminal_result->'provider_receipt'->>'latency_ms')::BIGINT
            IS DISTINCT FROM p_latency_ms
       OR p_billing_state IS DISTINCT FROM 'known'
       OR p_authoritative_billed_credit_microunits IS DISTINCT FROM p_credit_microunits
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_billing_mismatch'
            USING ERRCODE = '22023';
    END IF;
    SELECT * INTO reserve_event
      FROM public.research_lab_routing_budget_events_v2 candidate
     WHERE candidate.experiment_hash = p_experiment_hash
       AND candidate.reservation_id = budget_reservation->>'reservation_id'
       AND candidate.event_key = budget_reservation->>'event_key'
       AND candidate.event_type = 'reserve'
     ORDER BY candidate.created_at ASC, candidate.event_key ASC
     LIMIT 1;
    IF NOT FOUND
       OR reserve_event.binding_id IS DISTINCT FROM p_binding_id
       OR reserve_event.claim_key IS DISTINCT FROM call_grant_doc->>'claim_key'
       OR reserve_event.claim_generation
            IS DISTINCT FROM (call_grant_doc->>'claim_generation')::BIGINT
       OR reserve_event.credit_microunits < p_credit_microunits
       OR budget_reservation->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR budget_reservation->>'binding_id' IS DISTINCT FROM p_binding_id
       OR budget_reservation->>'claim_key' IS DISTINCT FROM reserve_event.claim_key
       OR budget_reservation->>'claim_generation'
            IS DISTINCT FROM reserve_event.claim_generation::TEXT
       OR budget_reservation->>'credit_microunits'
            IS DISTINCT FROM reserve_event.credit_microunits::TEXT
       OR budget_reservation->'lease_expires_at'
            IS DISTINCT FROM pg_catalog.to_jsonb(reserve_event.lease_expires_at)
       OR budget_reservation->>'response_hash' NOT IN (
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.research_lab.routing_budget_reservation_result.v3',
                    'reserved', TRUE,
                    'idempotent', FALSE,
                    'reservation_id', reserve_event.reservation_id,
                    'event_key', reserve_event.event_key,
                    'experiment_hash', reserve_event.experiment_hash,
                    'binding_id', reserve_event.binding_id,
                    'claim_key', reserve_event.claim_key,
                    'claim_generation', reserve_event.claim_generation,
                    'credit_microunits', reserve_event.credit_microunits,
                    'lease_expires_at', reserve_event.lease_expires_at
                )
            ),
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.research_lab.routing_budget_reservation_result.v3',
                    'reserved', TRUE,
                    'idempotent', TRUE,
                    'reservation_id', reserve_event.reservation_id,
                    'event_key', reserve_event.event_key,
                    'experiment_hash', reserve_event.experiment_hash,
                    'binding_id', reserve_event.binding_id,
                    'claim_key', reserve_event.claim_key,
                    'claim_generation', reserve_event.claim_generation,
                    'credit_microunits', reserve_event.credit_microunits,
                    'lease_expires_at', reserve_event.lease_expires_at
                )
            )
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_reservation_proof_mismatch'
            USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_attested_execution_receipts_v2 receipt
          JOIN public.research_lab_attested_boot_identities_v2 signer
            ON signer.boot_identity_hash = receipt.boot_identity_hash
         WHERE receipt.receipt_hash = p_authorization_proof_hash
           AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
           AND receipt.role = 'gateway_scoring'
           AND receipt.purpose = 'research_lab.routing_provider_evidence.v2'
           AND receipt.receipt_status = 'succeeded'
           AND receipt.job_id = authorization_job_id
           AND receipt.input_root = p_authorization_request_hash
           AND receipt.output_root = grant_result->>'output_root'
           AND receipt.receipt_doc = authorization_doc
           AND receipt.receipt_doc->'parent_receipt_hashes' =
                pg_catalog.jsonb_build_array(p_protected_release_receipt_hash, model_observation_hash)
           AND receipt.enclave_pubkey = signer.signing_pubkey
    ) OR NOT EXISTS (
        SELECT 1
          FROM public.research_lab_attested_execution_receipts_v2 receipt
          JOIN public.research_lab_attested_boot_identities_v2 signer
            ON signer.boot_identity_hash = receipt.boot_identity_hash
         WHERE receipt.receipt_hash = p_protected_release_receipt_hash
           AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
           AND receipt.role = 'gateway_scoring'
           AND receipt.purpose = 'research_lab.routing_provider_evidence.v2'
           AND receipt.receipt_status = 'succeeded'
           AND receipt.job_id = admission_job_id
           AND receipt.receipt_doc = protected_doc
           AND receipt.enclave_pubkey = signer.signing_pubkey
    ) OR NOT EXISTS (
        SELECT 1
          FROM public.research_lab_attested_execution_receipts_v2 receipt
          JOIN public.research_lab_attested_boot_identities_v2 signer
            ON signer.boot_identity_hash = receipt.boot_identity_hash
         WHERE receipt.receipt_hash = p_terminal_receipt_hash
           AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
           AND receipt.role = 'gateway_scoring'
           AND receipt.purpose = 'research_lab.routing_provider_evidence.v2'
           AND receipt.receipt_status = 'succeeded'
           AND receipt.job_id = terminal_job_id
           AND receipt.input_root = p_terminal_request_hash
           AND receipt.output_root = p_terminal_result_hash
           AND receipt.receipt_doc = terminal_doc
           AND receipt.receipt_doc->'parent_receipt_hashes' =
                pg_catalog.jsonb_build_array(p_authorization_proof_hash)
           AND receipt.enclave_pubkey = signer.signing_pubkey
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_receipt_missing' USING ERRCODE = '23503';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_attested_execution_receipts_v2 authorization_receipt
          JOIN public.research_lab_attested_execution_receipts_v2 protected_receipt
            ON protected_receipt.receipt_hash = p_protected_release_receipt_hash
          JOIN public.research_lab_attested_execution_receipts_v2 terminal_receipt
            ON terminal_receipt.receipt_hash = p_terminal_receipt_hash
         WHERE authorization_receipt.receipt_hash = p_authorization_proof_hash
           AND authorization_receipt.boot_identity_hash = protected_receipt.boot_identity_hash
           AND authorization_receipt.boot_identity_hash = terminal_receipt.boot_identity_hash
           AND authorization_receipt.enclave_pubkey = protected_receipt.enclave_pubkey
           AND authorization_receipt.enclave_pubkey = terminal_receipt.enclave_pubkey
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_receipt_signer_mismatch'
            USING ERRCODE = '23503';
    END IF;
END;
$receipt_chain_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_provider_attempt_v3(
    p_attempt_key TEXT,
    p_experiment_hash TEXT,
    p_provider_receipt_ref TEXT,
    p_binding_id TEXT,
    p_tool_id TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_reservation_id TEXT,
    p_action_id TEXT,
    p_binding_catalog_manifest_hash TEXT,
    p_authorization_hash TEXT,
    p_authorization_request_hash TEXT,
    p_authorization_proof_hash TEXT,
    p_request_body_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_request_fingerprint TEXT,
    p_outcome TEXT,
    p_credit_microunits BIGINT,
    p_latency_ms BIGINT,
    p_execution_mode TEXT,
    p_billing_state TEXT,
    p_authoritative_billed_credit_microunits BIGINT,
    p_terminal_receipt_hash TEXT,
    p_protected_release_receipt_hash TEXT,
    p_admission_bundle_hash TEXT,
    p_terminal_request_hash TEXT,
    p_terminal_result_hash TEXT,
    p_terminal_provider_record_hash TEXT,
    p_terminal_billing_projection_hash TEXT,
    p_attempt_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $append_attempt_v3$
DECLARE
    existing public.research_lab_routing_provider_attempts_v2%ROWTYPE;
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    expected_claim_fence_hash TEXT;
    budget_reservation JSONB := p_attempt_doc->'terminal_result'->'budget_reservation';
    attempt_identity_lock BIGINT;
    receipt_identity_lock BIGINT;
BEGIN
    IF p_attempt_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_provider_receipt_ref !~ '^provider_receipt:[0-9a-f]{16}$'
       OR p_binding_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_tool_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_unit_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_action_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_binding_catalog_manifest_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_authorization_proof_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_request_body_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR p_request_fingerprint !~ '^sha256:[0-9a-f]{64}$'
       OR p_outcome NOT IN ('verified', 'rejected', 'source_miss', 'adapter_failure')
       OR p_credit_microunits IS NULL OR p_credit_microunits < 0 OR p_credit_microunits > 10000000
       OR p_latency_ms IS NULL OR p_latency_ms < 0 OR p_latency_ms > 900000
       OR p_execution_mode NOT IN ('fixture', 'replay', 'measured_lab')
       OR p_billing_state NOT IN ('known', 'uncertain')
       OR p_terminal_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_protected_release_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_admission_bundle_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_result_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_provider_record_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_terminal_billing_projection_hash !~ '^sha256:[0-9a-f]{64}$'
       OR (p_billing_state = 'known' AND (
           p_authoritative_billed_credit_microunits IS NULL
           OR p_authoritative_billed_credit_microunits < 0
           OR p_authoritative_billed_credit_microunits > 10000000
       ))
       OR (p_billing_state = 'uncertain' AND p_authoritative_billed_credit_microunits IS NOT NULL)
       OR (p_outcome = 'adapter_failure' AND p_credit_microunits <> 0)
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_invalid_arguments' USING ERRCODE = '22023';
    END IF;
    IF p_outcome <> 'adapter_failure'
       AND (p_billing_state IS DISTINCT FROM 'known'
         OR p_authoritative_billed_credit_microunits IS DISTINCT FROM p_credit_microunits)
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_billing_state_invalid' USING ERRCODE = '22023';
    END IF;
    expected_claim_fence_hash := public.research_lab_routing_jsonb_hash_v2(
        pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_claim_fence.v3',
            'experiment_hash', p_experiment_hash,
            'claim_key', p_claim_key,
            'claim_generation', p_claim_generation
        )
    );
    PERFORM public.research_lab_routing_reject_secret_doc_v2(p_attempt_doc, 'routing provider attempt v3');
    IF p_attempt_doc->>'binding_id' IS DISTINCT FROM p_binding_id
       OR p_attempt_doc->>'tool_id' IS DISTINCT FROM p_tool_id
       OR p_attempt_doc->>'action_id' IS DISTINCT FROM p_action_id
       OR p_attempt_doc->>'binding_catalog_manifest_hash'
            IS DISTINCT FROM p_binding_catalog_manifest_hash
       OR p_attempt_doc->>'call_grant_hash' IS DISTINCT FROM p_authorization_hash
       OR p_attempt_doc->>'call_grant_proof_hash'
            IS DISTINCT FROM p_authorization_proof_hash
       OR p_attempt_doc->>'authorization_request_hash'
            IS DISTINCT FROM p_authorization_request_hash
       OR p_attempt_doc->>'request_body_hash' IS DISTINCT FROM p_request_body_hash
       OR p_attempt_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR p_attempt_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR p_attempt_doc->>'reservation_id' IS DISTINCT FROM p_reservation_id
       OR p_attempt_doc->>'request_fingerprint' IS DISTINCT FROM p_request_fingerprint
       OR p_attempt_doc->>'execution_mode' IS DISTINCT FROM p_execution_mode
       OR p_attempt_doc->>'terminal_request_hash' IS DISTINCT FROM p_terminal_request_hash
       OR p_attempt_doc->'call_grant'->>'claim_key' IS DISTINCT FROM p_claim_key
       OR p_attempt_doc->'call_grant'->>'claim_generation'
            IS DISTINCT FROM p_claim_generation::TEXT
       OR p_attempt_doc->'call_grant'->>'claim_fence_hash'
            IS DISTINCT FROM expected_claim_fence_hash
       OR p_attempt_doc->'call_grant_result'->>'claim_generation'
            IS DISTINCT FROM p_claim_generation::TEXT
       OR p_attempt_doc->'call_grant_result'->>'claim_fence_hash'
            IS DISTINCT FROM expected_claim_fence_hash
       OR p_attempt_doc->'provider_receipt'->>'receipt_ref'
            IS DISTINCT FROM p_provider_receipt_ref
       OR p_attempt_doc->'provider_receipt'->>'binding_id' IS DISTINCT FROM p_binding_id
       OR p_attempt_doc->'provider_receipt'->>'tool_id' IS DISTINCT FROM p_tool_id
       OR p_attempt_doc->'provider_receipt'->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR p_attempt_doc->'provider_receipt'->>'request_fingerprint'
            IS DISTINCT FROM p_request_fingerprint
       OR p_attempt_doc->'provider_receipt'->>'outcome' IS DISTINCT FROM p_outcome
       OR p_attempt_doc->'provider_receipt'->>'credit_microunits'
            IS DISTINCT FROM p_credit_microunits::TEXT
       OR p_attempt_doc->'provider_receipt'->>'latency_ms'
            IS DISTINCT FROM p_latency_ms::TEXT
       OR p_attempt_doc->'provider_receipt'->>'execution_mode'
            IS DISTINCT FROM p_execution_mode
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_document_binding_mismatch'
            USING ERRCODE = '22023';
    END IF;
    -- Idempotency is intentionally checked before claim/lease validation.  A
    -- byte-exact terminal replay is safe after a worker lease has expired;
    -- any changed field still fails closed.
    -- The provider-attempt and pre-dispatch-failure ledgers have separate
    -- unique indexes.  Lock both shared identities in a deterministic order
    -- before either ledger checks the other, so cross-ledger replays cannot
    -- pass their checks concurrently and both insert.
    attempt_identity_lock := pg_catalog.hashtextextended(p_attempt_key, 0);
    receipt_identity_lock := pg_catalog.hashtextextended(p_provider_receipt_ref, 0);
    IF attempt_identity_lock <= receipt_identity_lock THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(attempt_identity_lock);
        IF receipt_identity_lock <> attempt_identity_lock THEN
            PERFORM pg_catalog.pg_advisory_xact_lock(receipt_identity_lock);
        END IF;
    ELSE
        PERFORM pg_catalog.pg_advisory_xact_lock(receipt_identity_lock);
        PERFORM pg_catalog.pg_advisory_xact_lock(attempt_identity_lock);
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_routing_provider_attempts_v2
     WHERE attempt_key = p_attempt_key;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.provider_receipt_ref IS DISTINCT FROM p_provider_receipt_ref
           OR existing.binding_id IS DISTINCT FROM p_binding_id
           OR existing.tool_id IS DISTINCT FROM p_tool_id
           OR existing.variant_id IS DISTINCT FROM p_variant_id
           OR existing.unit_ref IS DISTINCT FROM p_unit_ref
           OR existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.action_id IS DISTINCT FROM p_action_id
           OR existing.binding_catalog_manifest_hash IS DISTINCT FROM p_binding_catalog_manifest_hash
           OR existing.authorization_hash IS DISTINCT FROM p_authorization_hash
           OR existing.authorization_proof_hash IS DISTINCT FROM p_authorization_proof_hash
           OR existing.request_body_hash IS DISTINCT FROM p_request_body_hash
           OR existing.authorization_request_hash IS DISTINCT FROM p_authorization_request_hash
           OR existing.terminal_receipt_hash IS DISTINCT FROM p_terminal_receipt_hash
           OR existing.protected_release_receipt_hash IS DISTINCT FROM p_protected_release_receipt_hash
           OR existing.admission_bundle_hash IS DISTINCT FROM p_admission_bundle_hash
           OR existing.terminal_request_hash IS DISTINCT FROM p_terminal_request_hash
           OR existing.terminal_result_hash IS DISTINCT FROM p_terminal_result_hash
           OR existing.terminal_provider_record_hash IS DISTINCT FROM p_terminal_provider_record_hash
           OR existing.terminal_billing_projection_hash IS DISTINCT FROM p_terminal_billing_projection_hash
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.request_fingerprint IS DISTINCT FROM p_request_fingerprint
           OR existing.outcome IS DISTINCT FROM p_outcome
           OR existing.credit_microunits IS DISTINCT FROM p_credit_microunits
           OR existing.billing_state IS DISTINCT FROM p_billing_state
           OR existing.authoritative_billed_credit_microunits IS DISTINCT FROM p_authoritative_billed_credit_microunits
           OR existing.latency_ms IS DISTINCT FROM p_latency_ms
           OR existing.execution_mode IS DISTINCT FROM p_execution_mode
           OR existing.attempt_doc IS DISTINCT FROM p_attempt_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_attempt_v3_conflict' USING ERRCODE = '23505';
        END IF;
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v3(
            existing.experiment_hash, existing.binding_id, existing.tool_id,
            existing.variant_id, existing.unit_ref, existing.action_id,
            existing.authorization_hash, existing.authorization_request_hash,
            existing.authorization_proof_hash, existing.terminal_receipt_hash,
            existing.protected_release_receipt_hash, existing.admission_bundle_hash,
            existing.terminal_request_hash, existing.terminal_result_hash,
            existing.terminal_provider_record_hash,
            existing.terminal_billing_projection_hash, existing.outcome,
            existing.credit_microunits, existing.latency_ms, existing.billing_state,
            existing.authoritative_billed_credit_microunits, existing.attempt_doc
        );
        RETURN pg_catalog.jsonb_build_object(
            'attempt_key', existing.attempt_key, 'idempotent', TRUE
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    SELECT * INTO reserve_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE experiment_hash = p_experiment_hash
       AND reservation_id = p_reservation_id
       AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC
     LIMIT 1;
    IF NOT FOUND
       OR reserve_event.binding_id IS DISTINCT FROM p_binding_id
       OR reserve_event.claim_key IS DISTINCT FROM p_claim_key
       OR reserve_event.claim_generation IS DISTINCT FROM p_claim_generation
       OR reserve_event.event_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR reserve_event.event_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR reserve_event.event_doc->>'request_fingerprint' IS DISTINCT FROM p_request_fingerprint
       OR reserve_event.event_doc->>'action_id' IS DISTINCT FROM p_action_id
       OR reserve_event.event_doc->>'binding_catalog_manifest_hash'
            IS DISTINCT FROM p_binding_catalog_manifest_hash
       OR reserve_event.event_doc->>'call_grant_hash'
            IS DISTINCT FROM p_authorization_hash
       OR reserve_event.event_doc->>'request_body_hash'
            IS DISTINCT FROM p_request_body_hash
       OR reserve_event.credit_microunits < p_credit_microunits
       OR budget_reservation->>'schema_version' IS DISTINCT FROM
            'leadpoet.research_lab.routing_budget_reservation_proof.v3'
       OR budget_reservation->>'reservation_id' IS DISTINCT FROM p_reservation_id
       OR budget_reservation->>'event_key' IS DISTINCT FROM reserve_event.event_key
       OR budget_reservation->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR budget_reservation->>'binding_id' IS DISTINCT FROM p_binding_id
       OR budget_reservation->>'claim_key' IS DISTINCT FROM p_claim_key
       OR budget_reservation->>'claim_generation'
            IS DISTINCT FROM p_claim_generation::TEXT
       OR budget_reservation->>'credit_microunits'
            IS DISTINCT FROM reserve_event.credit_microunits::TEXT
       OR budget_reservation->'lease_expires_at'
            IS DISTINCT FROM pg_catalog.to_jsonb(reserve_event.lease_expires_at)
       OR budget_reservation->>'response_hash' NOT IN (
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.research_lab.routing_budget_reservation_result.v3',
                    'reserved', TRUE,
                    'idempotent', FALSE,
                    'reservation_id', reserve_event.reservation_id,
                    'event_key', reserve_event.event_key,
                    'experiment_hash', reserve_event.experiment_hash,
                    'binding_id', reserve_event.binding_id,
                    'claim_key', reserve_event.claim_key,
                    'claim_generation', reserve_event.claim_generation,
                    'credit_microunits', reserve_event.credit_microunits,
                    'lease_expires_at', reserve_event.lease_expires_at
                )
            ),
            public.research_lab_routing_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'schema_version', 'leadpoet.research_lab.routing_budget_reservation_result.v3',
                    'reserved', TRUE,
                    'idempotent', TRUE,
                    'reservation_id', reserve_event.reservation_id,
                    'event_key', reserve_event.event_key,
                    'experiment_hash', reserve_event.experiment_hash,
                    'binding_id', reserve_event.binding_id,
                    'claim_key', reserve_event.claim_key,
                    'claim_generation', reserve_event.claim_generation,
                    'credit_microunits', reserve_event.credit_microunits,
                    'lease_expires_at', reserve_event.lease_expires_at
                )
            )
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_reservation_mismatch' USING ERRCODE = '23503';
    END IF;
    SELECT * INTO latest_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
     ORDER BY created_at DESC, event_key DESC
     LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_reservation_closed' USING ERRCODE = '23505';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiments_v2 experiment
          CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(experiment.spec_doc->'provider_bindings') AS binding(value)
          CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(experiment.spec_doc->'variants') AS variant(value)
         WHERE experiment.experiment_hash = p_experiment_hash
           AND experiment.receipt_execution_mode = p_execution_mode
           AND experiment.execution_envelope_doc->>'binding_catalog_manifest_hash'
                = p_binding_catalog_manifest_hash
           AND binding.value->>'binding_id' = p_binding_id
           AND binding.value->>'tool_id' = p_tool_id
           AND variant.value->>'variant_id' = p_variant_id
           AND p_binding_id = ANY (
               ARRAY(SELECT pg_catalog.jsonb_array_elements_text(variant.value->'binding_ids'))
           )
           AND EXISTS (
               SELECT 1
                 FROM pg_catalog.jsonb_array_elements(experiment.execution_envelope_doc->'bindings') AS runtime_binding(value)
                WHERE runtime_binding.value->>'binding_id' = p_binding_id
                  AND runtime_binding.value->>'provider_id' = binding.value->>'provider_id'
                  AND runtime_binding.value->>'tool_id' = p_tool_id
                  AND runtime_binding.value->>'binding_manifest_hash' = binding.value->>'manifest_hash'
                  AND runtime_binding.value->>'action_id' = p_action_id
           )
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_attempt_v3_binding_not_declared'
            USING ERRCODE = '23503';
    END IF;
    PERFORM public.research_lab_routing_assert_provider_receipt_chain_v3(
        p_experiment_hash, p_binding_id, p_tool_id, p_variant_id, p_unit_ref,
        p_action_id, p_authorization_hash, p_authorization_request_hash,
        p_authorization_proof_hash, p_terminal_receipt_hash,
        p_protected_release_receipt_hash, p_admission_bundle_hash,
        p_terminal_request_hash, p_terminal_result_hash,
        p_terminal_provider_record_hash, p_terminal_billing_projection_hash,
        p_outcome, p_credit_microunits, p_latency_ms, p_billing_state,
        p_authoritative_billed_credit_microunits, p_attempt_doc
    );
    INSERT INTO public.research_lab_routing_provider_attempts_v2 (
        attempt_key, experiment_hash, provider_receipt_ref, binding_id, tool_id,
        variant_id, unit_ref, reservation_id, action_id,
        binding_catalog_manifest_hash, authorization_hash,
        authorization_proof_hash, request_body_hash, authorization_request_hash,
        terminal_receipt_hash, protected_release_receipt_hash,
        admission_bundle_hash, terminal_request_hash, terminal_result_hash,
        terminal_provider_record_hash, terminal_billing_projection_hash,
        claim_key, claim_generation, request_fingerprint, outcome,
        credit_microunits, billing_state, authoritative_billed_credit_microunits,
        latency_ms, execution_mode, attempt_doc
    ) VALUES (
        p_attempt_key, p_experiment_hash, p_provider_receipt_ref, p_binding_id,
        p_tool_id, p_variant_id, p_unit_ref, p_reservation_id, p_action_id,
        p_binding_catalog_manifest_hash, p_authorization_hash,
        p_authorization_proof_hash, p_request_body_hash, p_authorization_request_hash,
        p_terminal_receipt_hash, p_protected_release_receipt_hash,
        p_admission_bundle_hash, p_terminal_request_hash, p_terminal_result_hash,
        p_terminal_provider_record_hash, p_terminal_billing_projection_hash,
        p_claim_key, p_claim_generation, p_request_fingerprint, p_outcome,
        p_credit_microunits, p_billing_state, p_authoritative_billed_credit_microunits,
        p_latency_ms, p_execution_mode, p_attempt_doc
    );
    RETURN pg_catalog.jsonb_build_object('attempt_key', p_attempt_key, 'idempotent', FALSE);
END;
$append_attempt_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_decision_receipt_v3(
    p_receipt_id TEXT,
    p_experiment_hash TEXT,
    p_variant_id TEXT,
    p_unit_ref TEXT,
    p_plan_hash TEXT,
    p_route_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_decision_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $decision_v3$
DECLARE
    existing public.research_lab_routing_decision_receipts_v2%ROWTYPE;
BEGIN
    IF p_receipt_id !~ '^routing_decision:[0-9a-f]{16}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_unit_ref !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_plan_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_route_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_decision_doc) IS DISTINCT FROM 'object'
       OR p_decision_doc->>'receipt_id' IS DISTINCT FROM p_receipt_id
    THEN
        RAISE EXCEPTION 'research_lab_routing_decision_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_decision_doc, 'routing decision v3'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_receipt_id, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_decision_receipts_v2
     WHERE receipt_id = p_receipt_id;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.variant_id IS DISTINCT FROM p_variant_id
           OR existing.unit_ref IS DISTINCT FROM p_unit_ref
           OR existing.plan_hash IS DISTINCT FROM p_plan_hash
           OR existing.route_hash IS DISTINCT FROM p_route_hash
           OR existing.decision_doc IS DISTINCT FROM p_decision_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_decision_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'receipt_id', p_receipt_id, 'idempotent', TRUE
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiments_v2 experiment
          CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
              experiment.spec_doc->'variants'
          ) variant(value)
         WHERE experiment.experiment_hash = p_experiment_hash
           AND variant.value->>'variant_id' = p_variant_id
    ) OR p_decision_doc->>'variant_id' IS DISTINCT FROM p_variant_id
       OR p_decision_doc->>'unit_ref' IS DISTINCT FROM p_unit_ref
       OR p_decision_doc->>'plan_hash' IS DISTINCT FROM p_plan_hash
       OR p_decision_doc->>'route_hash' IS DISTINCT FROM p_route_hash
       OR pg_catalog.jsonb_typeof(p_decision_doc->'provider_receipt_refs')
            IS DISTINCT FROM 'array'
       OR EXISTS (
            SELECT 1
              FROM pg_catalog.jsonb_array_elements_text(
                  p_decision_doc->'provider_receipt_refs'
              ) ref(provider_receipt_ref)
             WHERE NOT EXISTS (
                 SELECT 1
                   FROM public.research_lab_routing_provider_attempts_v2 attempt
                  WHERE attempt.experiment_hash = p_experiment_hash
                    AND attempt.provider_receipt_ref = ref.provider_receipt_ref
             )
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_decision_v3_spec_or_document_mismatch'
            USING ERRCODE = '23503';
    END IF;
    INSERT INTO public.research_lab_routing_decision_receipts_v2 (
        receipt_id, experiment_hash, variant_id, unit_ref, claim_key,
        claim_generation, plan_hash, route_hash, decision_doc
    ) VALUES (
        p_receipt_id, p_experiment_hash, p_variant_id, p_unit_ref, p_claim_key,
        p_claim_generation, p_plan_hash, p_route_hash, p_decision_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'receipt_id', p_receipt_id, 'idempotent', FALSE
    );
END;
$decision_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_append_evaluation_v3(
    p_receipt_id TEXT,
    p_experiment_hash TEXT,
    p_evaluation_hash TEXT,
    p_selected_variant_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_evaluation_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $evaluation_v3$
DECLARE
    existing public.research_lab_routing_evaluation_receipts_v2%ROWTYPE;
    experiment public.research_lab_routing_experiments_v2%ROWTYPE;
BEGIN
    IF p_receipt_id !~ '^routing_evaluation_v2:[0-9a-f]{16}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_evaluation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_evaluation_hash IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(p_evaluation_doc)
       OR (p_selected_variant_id <> 'unselected'
           AND p_selected_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$')
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_evaluation_doc) IS DISTINCT FROM 'object'
       OR p_evaluation_doc->>'receipt_id' IS DISTINCT FROM p_receipt_id
       OR p_evaluation_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR (p_evaluation_doc->>'selected_variant_id' IS DISTINCT FROM p_selected_variant_id
           AND NOT (
               p_selected_variant_id = 'unselected'
               AND p_evaluation_doc->>'selected_variant_id' = ''
           ))
       OR pg_catalog.jsonb_typeof(p_evaluation_doc->'decision_receipt_refs')
            IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(p_evaluation_doc->'provider_receipt_refs')
            IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(p_evaluation_doc->'variants')
            IS DISTINCT FROM 'array'
    THEN
        RAISE EXCEPTION 'research_lab_routing_evaluation_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_evaluation_doc, 'routing evaluation v3'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_receipt_id, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_evaluation_receipts_v2
     WHERE receipt_id = p_receipt_id;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.evaluation_hash IS DISTINCT FROM p_evaluation_hash
           OR existing.selected_variant_id IS DISTINCT FROM p_selected_variant_id
           OR existing.evaluation_doc IS DISTINCT FROM p_evaluation_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_evaluation_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'receipt_id', p_receipt_id, 'idempotent', TRUE
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    SELECT * INTO experiment
      FROM public.research_lab_routing_experiments_v2
     WHERE experiment_hash = p_experiment_hash;
    IF NOT FOUND
       OR pg_catalog.jsonb_array_length(p_evaluation_doc->'decision_receipt_refs') = 0
       OR pg_catalog.jsonb_array_length(p_evaluation_doc->'provider_receipt_refs') = 0
       OR (p_selected_variant_id <> 'unselected' AND (
           NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.jsonb_array_elements(
                     experiment.spec_doc->'variants'
                 ) variant(value)
                WHERE variant.value->>'variant_id' = p_selected_variant_id
           ) OR NOT EXISTS (
               SELECT 1
                 FROM pg_catalog.jsonb_array_elements(
                     p_evaluation_doc->'variants'
                 ) variant(value)
                WHERE variant.value->>'variant_id' = p_selected_variant_id
           )
       ))
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements_text(
                 p_evaluation_doc->'decision_receipt_refs'
             ) ref(receipt_id)
            WHERE NOT EXISTS (
                SELECT 1
                  FROM public.research_lab_routing_decision_receipts_v2 decision
                 WHERE decision.experiment_hash = p_experiment_hash
                   AND decision.receipt_id = ref.receipt_id
            )
       ) OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements_text(
                 p_evaluation_doc->'provider_receipt_refs'
             ) ref(provider_receipt_ref)
            WHERE NOT EXISTS (
                SELECT 1
                  FROM public.research_lab_routing_provider_attempts_v2 attempt
                 WHERE attempt.experiment_hash = p_experiment_hash
                   AND attempt.provider_receipt_ref = ref.provider_receipt_ref
            )
       ) OR EXISTS (
           (SELECT decision.receipt_id
              FROM public.research_lab_routing_decision_receipts_v2 decision
             WHERE decision.experiment_hash = p_experiment_hash)
           EXCEPT
           (SELECT ref.receipt_id
              FROM pg_catalog.jsonb_array_elements_text(
                  p_evaluation_doc->'decision_receipt_refs'
              ) ref(receipt_id))
       ) OR EXISTS (
           (SELECT attempt.provider_receipt_ref
              FROM public.research_lab_routing_provider_attempts_v2 attempt
             WHERE attempt.experiment_hash = p_experiment_hash)
           EXCEPT
           (SELECT ref.provider_receipt_ref
              FROM pg_catalog.jsonb_array_elements_text(
                  p_evaluation_doc->'provider_receipt_refs'
              ) ref(provider_receipt_ref))
       ) OR (
           SELECT count(*)
             FROM pg_catalog.jsonb_array_elements_text(
                 p_evaluation_doc->'decision_receipt_refs'
             ) ref(receipt_id)
       ) <> (
           SELECT count(DISTINCT ref.receipt_id)
             FROM pg_catalog.jsonb_array_elements_text(
                 p_evaluation_doc->'decision_receipt_refs'
             ) ref(receipt_id)
       ) OR (
           SELECT count(*)
             FROM pg_catalog.jsonb_array_elements_text(
                 p_evaluation_doc->'provider_receipt_refs'
             ) ref(provider_receipt_ref)
       ) <> (
           SELECT count(DISTINCT ref.provider_receipt_ref)
             FROM pg_catalog.jsonb_array_elements_text(
                 p_evaluation_doc->'provider_receipt_refs'
             ) ref(provider_receipt_ref)
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_evaluation_v3_receipt_set_incomplete'
            USING ERRCODE = '23503';
    END IF;
    INSERT INTO public.research_lab_routing_evaluation_receipts_v2 (
        receipt_id, experiment_hash, evaluation_hash, selected_variant_id,
        claim_key, claim_generation, evaluation_doc
    ) VALUES (
        p_receipt_id, p_experiment_hash, p_evaluation_hash, p_selected_variant_id,
        p_claim_key, p_claim_generation, p_evaluation_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'receipt_id', p_receipt_id, 'idempotent', FALSE
    );
END;
$evaluation_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_reserve_budget_v3(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_experiment_hash TEXT,
    p_binding_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_credit_microunits BIGINT,
    p_lease_seconds INTEGER,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $reserve_v3$
DECLARE
    existing public.research_lab_routing_budget_events_v2%ROWTYPE;
    experiment public.research_lab_routing_experiments_v2%ROWTYPE;
    total_budget BIGINT;
    binding_budget BIGINT;
    consumed BIGINT;
    binding_consumed BIGINT;
    expiry TIMESTAMPTZ;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_binding_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR p_credit_microunits IS NULL OR p_credit_microunits < 0
       OR p_credit_microunits > 10000000
       OR p_lease_seconds IS NULL OR p_lease_seconds < 1 OR p_lease_seconds > 3600
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
       OR p_event_doc->>'reservation_id' IS DISTINCT FROM p_reservation_id
       OR p_event_doc->>'binding_id' IS DISTINCT FROM p_binding_id
       OR p_event_doc->>'unit_ref' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'variant_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'request_fingerprint' !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_doc->>'action_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'tool_id' !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_doc->>'binding_catalog_manifest_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_doc->>'call_grant_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_event_doc->>'request_body_hash' !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_event_doc, 'routing budget reserve v3'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_event_key, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_budget_events_v2
     WHERE event_key = p_event_key;
    IF FOUND THEN
        IF existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.binding_id IS DISTINCT FROM p_binding_id
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_type IS DISTINCT FROM 'reserve'
           OR existing.credit_microunits IS DISTINCT FROM p_credit_microunits
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_reserve_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'schema_version', 'leadpoet.research_lab.routing_budget_reservation_result.v3',
            'reserved', TRUE,
            'idempotent', TRUE,
            'reservation_id', existing.reservation_id,
            'event_key', existing.event_key,
            'experiment_hash', existing.experiment_hash,
            'binding_id', existing.binding_id,
            'claim_key', existing.claim_key,
            'claim_generation', existing.claim_generation,
            'credit_microunits', existing.credit_microunits,
            'lease_expires_at', existing.lease_expires_at
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    SELECT * INTO experiment
      FROM public.research_lab_routing_experiments_v2
     WHERE experiment_hash = p_experiment_hash;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_v3_experiment_missing'
            USING ERRCODE = '23503';
    END IF;
    total_budget := coalesce(
        (experiment.spec_doc #>> '{credit_budget,total_credit_microunits}')::BIGINT,
        -1
    );
    binding_budget := coalesce(
        (experiment.spec_doc #>> ARRAY[
            'credit_budget', 'provider_credit_ceilings', p_binding_id
        ])::BIGINT,
        -1
    );
    IF experiment.allow_live_credit_spend IS NOT TRUE
       OR experiment.receipt_execution_mode IS DISTINCT FROM 'measured_lab'
       OR total_budget < 0 OR binding_budget < 0
       OR NOT EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements(
                 experiment.spec_doc->'provider_bindings'
             ) model_binding(value)
             JOIN pg_catalog.jsonb_array_elements(
                 experiment.execution_envelope_doc->'bindings'
             ) runtime_binding(value)
               ON runtime_binding.value->>'binding_id'
                    = model_binding.value->>'binding_id'
            WHERE model_binding.value->>'binding_id' = p_binding_id
              AND model_binding.value->>'tool_id' = p_event_doc->>'tool_id'
              AND runtime_binding.value->>'tool_id' = p_event_doc->>'tool_id'
              AND runtime_binding.value->>'provider_id'
                    = model_binding.value->>'provider_id'
              AND runtime_binding.value->>'binding_manifest_hash'
                    = model_binding.value->>'manifest_hash'
              AND runtime_binding.value->>'action_id' = p_event_doc->>'action_id'
              AND experiment.execution_envelope_doc->>'binding_catalog_manifest_hash'
                    = p_event_doc->>'binding_catalog_manifest_hash'
              AND coalesce(
                    (runtime_binding.value->>'credit_ceiling_microunits')::BIGINT,
                    -1
                  ) >= p_credit_microunits
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_v3_budget_contract_missing'
            USING ERRCODE = '22023';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM (
              SELECT DISTINCT ON (budget_event.reservation_id)
                  budget_event.event_type, budget_event.lease_expires_at
                FROM public.research_lab_routing_budget_events_v2 budget_event
               WHERE budget_event.experiment_hash = p_experiment_hash
               ORDER BY budget_event.reservation_id,
                   budget_event.created_at DESC, budget_event.event_key DESC
          ) budget_head
         WHERE budget_head.event_type = 'reserve'
           AND budget_head.lease_expires_at <= pg_catalog.clock_timestamp()
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_v3_expired_unknown_exists'
            USING ERRCODE = '23505';
    END IF;
    SELECT coalesce(sum(budget_head.credit_microunits), 0) INTO consumed
      FROM (
          SELECT DISTINCT ON (reservation_id) *
            FROM public.research_lab_routing_budget_events_v2
           WHERE experiment_hash = p_experiment_hash
           ORDER BY reservation_id, created_at DESC, event_key DESC
      ) budget_head
     WHERE budget_head.event_type IN ('settle', 'uncertain', 'recover')
        OR (budget_head.event_type = 'reserve'
            AND budget_head.lease_expires_at > pg_catalog.clock_timestamp());
    SELECT coalesce(sum(budget_head.credit_microunits), 0) INTO binding_consumed
      FROM (
          SELECT DISTINCT ON (reservation_id) *
            FROM public.research_lab_routing_budget_events_v2
           WHERE experiment_hash = p_experiment_hash
             AND binding_id = p_binding_id
           ORDER BY reservation_id, created_at DESC, event_key DESC
      ) budget_head
     WHERE budget_head.event_type IN ('settle', 'uncertain', 'recover')
        OR (budget_head.event_type = 'reserve'
            AND budget_head.lease_expires_at > pg_catalog.clock_timestamp());
    IF consumed + p_credit_microunits > total_budget
       OR binding_consumed + p_credit_microunits > binding_budget
    THEN
        RAISE EXCEPTION 'research_lab_routing_reserve_v3_budget_exceeded'
            USING ERRCODE = '23505';
    END IF;
    expiry := pg_catalog.clock_timestamp()
        + pg_catalog.make_interval(secs => p_lease_seconds);
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type, credit_microunits, lease_expires_at,
        event_doc
    ) VALUES (
        p_event_key, p_reservation_id, p_experiment_hash, p_binding_id,
        p_claim_key, p_claim_generation, 'reserve', p_credit_microunits,
        expiry, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_budget_reservation_result.v3',
        'reserved', TRUE,
        'idempotent', FALSE,
        'reservation_id', p_reservation_id,
        'event_key', p_event_key,
        'experiment_hash', p_experiment_hash,
        'binding_id', p_binding_id,
        'claim_key', p_claim_key,
        'claim_generation', p_claim_generation,
        'credit_microunits', p_credit_microunits,
        'lease_expires_at', expiry
    );
END;
$reserve_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_settle_budget_v3(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_attempt_key TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $settle_v3$
DECLARE
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    attempt public.research_lab_routing_provider_attempts_v2%ROWTYPE;
    existing public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_attempt_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
       OR p_event_doc->>'attempt_key' IS DISTINCT FROM p_attempt_key
       OR p_event_doc->>'billing_state' IS DISTINCT FROM 'known'
    THEN
        RAISE EXCEPTION 'research_lab_routing_settle_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_event_doc, 'routing budget settlement v3'
    );
    SELECT * INTO reserve_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
       AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC
     LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_settle_v3_reservation_missing'
            USING ERRCODE = '23503';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(reserve_event.experiment_hash, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_event_key, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_budget_events_v2
     WHERE event_key = p_event_key;
    IF FOUND THEN
        IF existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
           OR existing.binding_id IS DISTINCT FROM reserve_event.binding_id
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.attempt_key IS DISTINCT FROM p_attempt_key
           OR existing.event_type IS DISTINCT FROM 'settle'
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_settle_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'settled', TRUE, 'idempotent', TRUE,
            'credit_microunits', existing.credit_microunits
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        reserve_event.experiment_hash, p_claim_key, p_claim_generation
    );
    SELECT * INTO latest_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
     ORDER BY created_at DESC, event_key DESC
     LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
        RAISE EXCEPTION 'research_lab_routing_settle_v3_reservation_closed'
            USING ERRCODE = '23505';
    END IF;
    SELECT * INTO attempt
      FROM public.research_lab_routing_provider_attempts_v2
     WHERE attempt_key = p_attempt_key;
    IF NOT FOUND
       OR attempt.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
       OR attempt.binding_id IS DISTINCT FROM reserve_event.binding_id
       OR attempt.reservation_id IS DISTINCT FROM p_reservation_id
       OR attempt.unit_ref IS DISTINCT FROM reserve_event.event_doc->>'unit_ref'
       OR attempt.variant_id IS DISTINCT FROM reserve_event.event_doc->>'variant_id'
       OR attempt.request_fingerprint
            IS DISTINCT FROM reserve_event.event_doc->>'request_fingerprint'
       OR attempt.action_id IS DISTINCT FROM reserve_event.event_doc->>'action_id'
       OR attempt.claim_key IS DISTINCT FROM p_claim_key
       OR attempt.claim_generation IS DISTINCT FROM p_claim_generation
       OR attempt.billing_state IS DISTINCT FROM 'known'
       OR attempt.authoritative_billed_credit_microunits IS NULL
       OR attempt.authoritative_billed_credit_microunits
            > reserve_event.credit_microunits
       OR attempt.authorization_request_hash IS NULL
       OR attempt.terminal_request_hash IS NULL
       OR attempt.terminal_result_hash IS NULL
    THEN
        RAISE EXCEPTION 'research_lab_routing_settle_v3_attempt_mismatch'
            USING ERRCODE = '23503';
    END IF;
    PERFORM public.research_lab_routing_assert_provider_receipt_chain_v3(
        attempt.experiment_hash, attempt.binding_id, attempt.tool_id,
        attempt.variant_id, attempt.unit_ref, attempt.action_id,
        attempt.authorization_hash, attempt.authorization_request_hash,
        attempt.authorization_proof_hash, attempt.terminal_receipt_hash,
        attempt.protected_release_receipt_hash, attempt.admission_bundle_hash,
        attempt.terminal_request_hash, attempt.terminal_result_hash,
        attempt.terminal_provider_record_hash,
        attempt.terminal_billing_projection_hash, attempt.outcome,
        attempt.credit_microunits, attempt.latency_ms, attempt.billing_state,
        attempt.authoritative_billed_credit_microunits, attempt.attempt_doc
    );
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, attempt_key, event_type, credit_microunits, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, reserve_event.experiment_hash,
        reserve_event.binding_id, p_claim_key, p_claim_generation, p_attempt_key,
        'settle', attempt.authoritative_billed_credit_microunits, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'settled', TRUE, 'idempotent', FALSE,
        'credit_microunits', attempt.authoritative_billed_credit_microunits
    );
END;
$settle_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_mark_budget_uncertain_v3(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $uncertain_v3$
DECLARE
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    existing public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
       OR p_event_doc->>'billing_state' IS DISTINCT FROM 'uncertain'
    THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_event_doc, 'routing uncertain budget v3'
    );
    SELECT * INTO reserve_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
       AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC
     LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_v3_reservation_missing'
            USING ERRCODE = '23503';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(reserve_event.experiment_hash, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_event_key, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_budget_events_v2
     WHERE event_key = p_event_key;
    IF FOUND THEN
        IF existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
           OR existing.binding_id IS DISTINCT FROM reserve_event.binding_id
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_type IS DISTINCT FROM 'uncertain'
           OR existing.credit_microunits IS DISTINCT FROM reserve_event.credit_microunits
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_uncertain_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'uncertain', TRUE, 'idempotent', TRUE,
            'credit_microunits', existing.credit_microunits
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        reserve_event.experiment_hash, p_claim_key, p_claim_generation
    );
    SELECT * INTO latest_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
     ORDER BY created_at DESC, event_key DESC
     LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve' THEN
        RAISE EXCEPTION 'research_lab_routing_uncertain_v3_reservation_closed'
            USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type, credit_microunits, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, reserve_event.experiment_hash,
        reserve_event.binding_id, p_claim_key, p_claim_generation,
        'uncertain', reserve_event.credit_microunits, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'uncertain', TRUE, 'idempotent', FALSE,
        'credit_microunits', reserve_event.credit_microunits
    );
END;
$uncertain_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_recover_budget_v3(
    p_event_key TEXT,
    p_reservation_id TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $recover_budget_v3$
DECLARE
    reserve_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    latest_event public.research_lab_routing_budget_events_v2%ROWTYPE;
    existing public.research_lab_routing_budget_events_v2%ROWTYPE;
BEGIN
    IF p_event_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_reservation_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
       OR p_event_doc->>'billing_state' IS DISTINCT FROM 'uncertain'
    THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_event_doc, 'routing budget recovery v3'
    );
    SELECT * INTO reserve_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
       AND event_type = 'reserve'
     ORDER BY created_at ASC, event_key ASC
     LIMIT 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_v3_reservation_missing'
            USING ERRCODE = '23503';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(reserve_event.experiment_hash, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_event_key, 0)
    );
    SELECT * INTO existing
      FROM public.research_lab_routing_budget_events_v2
     WHERE event_key = p_event_key;
    IF FOUND THEN
        IF existing.reservation_id IS DISTINCT FROM p_reservation_id
           OR existing.experiment_hash IS DISTINCT FROM reserve_event.experiment_hash
           OR existing.binding_id IS DISTINCT FROM reserve_event.binding_id
           OR existing.claim_key IS DISTINCT FROM p_claim_key
           OR existing.claim_generation IS DISTINCT FROM p_claim_generation
           OR existing.event_type IS DISTINCT FROM 'recover'
           OR existing.credit_microunits IS DISTINCT FROM reserve_event.credit_microunits
           OR existing.event_doc IS DISTINCT FROM p_event_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_recover_budget_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'recovered', TRUE, 'idempotent', TRUE,
            'billing_state', 'uncertain',
            'credit_microunits', existing.credit_microunits
        );
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        reserve_event.experiment_hash, p_claim_key, p_claim_generation
    );
    SELECT * INTO latest_event
      FROM public.research_lab_routing_budget_events_v2
     WHERE reservation_id = p_reservation_id
     ORDER BY created_at DESC, event_key DESC
     LIMIT 1;
    IF latest_event.event_type IS DISTINCT FROM 'reserve'
       OR latest_event.lease_expires_at >= pg_catalog.clock_timestamp()
    THEN
        RAISE EXCEPTION 'research_lab_routing_recover_budget_v3_not_expired'
            USING ERRCODE = '23505';
    END IF;
    INSERT INTO public.research_lab_routing_budget_events_v2 (
        event_key, reservation_id, experiment_hash, binding_id, claim_key,
        claim_generation, event_type, credit_microunits, event_doc
    ) VALUES (
        p_event_key, p_reservation_id, reserve_event.experiment_hash,
        reserve_event.binding_id, p_claim_key, p_claim_generation,
        'recover', reserve_event.credit_microunits, p_event_doc
    );
    RETURN pg_catalog.jsonb_build_object(
        'recovered', TRUE, 'idempotent', FALSE,
        'billing_state', 'uncertain',
        'credit_microunits', reserve_event.credit_microunits
    );
END;
$recover_budget_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_list_expired_budget_reservations_v3(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $list_expired_budgets_v3$
DECLARE
    reservations JSONB;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
    THEN
        RAISE EXCEPTION 'research_lab_routing_list_expired_budgets_v3_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    WITH latest AS (
        SELECT DISTINCT ON (budget_event.reservation_id)
            budget_event.reservation_id,
            budget_event.binding_id,
            budget_event.claim_key AS reserve_claim_key,
            budget_event.claim_generation AS reserve_claim_generation,
            budget_event.credit_microunits,
            budget_event.event_type,
            budget_event.lease_expires_at
          FROM public.research_lab_routing_budget_events_v2 budget_event
         WHERE budget_event.experiment_hash = p_experiment_hash
         ORDER BY budget_event.reservation_id,
             budget_event.created_at DESC, budget_event.event_key DESC
    ), expired AS (
        SELECT * FROM latest
         WHERE event_type = 'reserve'
           AND lease_expires_at <= pg_catalog.clock_timestamp()
    )
    SELECT coalesce(
        pg_catalog.jsonb_agg(
            pg_catalog.jsonb_build_object(
                'reservation_id', expired.reservation_id,
                'binding_id', expired.binding_id,
                'credit_microunits', expired.credit_microunits,
                'dispatch_started', EXISTS (
                    SELECT 1
                      FROM public.research_lab_routing_experiment_events_v2 dispatch_event
                     WHERE dispatch_event.experiment_hash = p_experiment_hash
                       AND dispatch_event.event_type = 'provider_dispatch_started'
                       AND dispatch_event.claim_key = expired.reserve_claim_key
                       AND dispatch_event.claim_generation = expired.reserve_claim_generation
                       AND dispatch_event.event_doc->>'reservation_id'
                            = expired.reservation_id
                       AND dispatch_event.event_doc->>'binding_id'
                            = expired.binding_id
                )
            ) ORDER BY expired.reservation_id
        ), '[]'::JSONB
    ) INTO reservations
      FROM expired;
    RETURN pg_catalog.jsonb_build_object('reservations', reservations);
END;
$list_expired_budgets_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_list_unresolved_budget_reservations_v3(
    p_experiment_hash TEXT,
    p_claim_key TEXT,
    p_claim_generation BIGINT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $list_unresolved_budgets_v3$
DECLARE
    reservations JSONB;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_key !~ '^sha256:[0-9a-f]{64}$'
       OR p_claim_generation IS NULL OR p_claim_generation < 1
    THEN
        RAISE EXCEPTION 'research_lab_routing_list_unresolved_budgets_v3_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_assert_claim_v3(
        p_experiment_hash, p_claim_key, p_claim_generation
    );
    WITH latest AS (
        SELECT DISTINCT ON (budget_event.reservation_id)
            budget_event.reservation_id,
            budget_event.binding_id,
            budget_event.claim_key AS reserve_claim_key,
            budget_event.claim_generation AS reserve_claim_generation,
            budget_event.credit_microunits,
            budget_event.event_type,
            budget_event.lease_expires_at
          FROM public.research_lab_routing_budget_events_v2 budget_event
         WHERE budget_event.experiment_hash = p_experiment_hash
         ORDER BY budget_event.reservation_id,
             budget_event.created_at DESC, budget_event.event_key DESC
    ), unresolved AS (
        SELECT * FROM latest WHERE event_type <> 'settle'
    )
    SELECT coalesce(
        pg_catalog.jsonb_agg(
            pg_catalog.jsonb_build_object(
                'reservation_id', unresolved.reservation_id,
                'binding_id', unresolved.binding_id,
                'credit_microunits', unresolved.credit_microunits,
                'event_type', unresolved.event_type,
                'lease_expired', (
                    unresolved.event_type = 'reserve'
                    AND unresolved.lease_expires_at <= pg_catalog.clock_timestamp()
                ),
                'dispatch_started', EXISTS (
                    SELECT 1
                      FROM public.research_lab_routing_experiment_events_v2 dispatch_event
                     WHERE dispatch_event.experiment_hash = p_experiment_hash
                       AND dispatch_event.event_type = 'provider_dispatch_started'
                       AND dispatch_event.claim_key = unresolved.reserve_claim_key
                       AND dispatch_event.claim_generation = unresolved.reserve_claim_generation
                       AND dispatch_event.event_doc->>'reservation_id'
                            = unresolved.reservation_id
                       AND dispatch_event.event_doc->>'binding_id'
                            = unresolved.binding_id
                )
            ) ORDER BY unresolved.reservation_id
        ), '[]'::JSONB
    ) INTO reservations
      FROM unresolved;
    RETURN pg_catalog.jsonb_build_object('reservations', reservations);
END;
$list_unresolved_budgets_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_promotion_receipt_chain_v3(
    p_experiment_hash TEXT
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $promotion_chain_v3$
DECLARE
    attempt public.research_lab_routing_provider_attempts_v2%ROWTYPE;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_experiment_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    FOR attempt IN
        SELECT *
          FROM public.research_lab_routing_provider_attempts_v2
         WHERE experiment_hash = p_experiment_hash
         ORDER BY attempt_key
    LOOP
        IF attempt.outcome = 'adapter_failure'
           OR attempt.billing_state IS DISTINCT FROM 'known'
           OR attempt.authoritative_billed_credit_microunits
                IS DISTINCT FROM attempt.credit_microunits
           OR attempt.authorization_request_hash IS NULL
           OR attempt.terminal_request_hash IS NULL
           OR attempt.terminal_result_hash IS NULL
           OR attempt.terminal_receipt_hash IS NULL
           OR attempt.protected_release_receipt_hash IS NULL
           OR attempt.admission_bundle_hash IS NULL
           OR attempt.terminal_provider_record_hash IS NULL
           OR attempt.terminal_billing_projection_hash IS NULL
        THEN
            RAISE EXCEPTION 'research_lab_routing_promote_v3_receipt_chain_missing'
                USING ERRCODE = '23503';
        END IF;
        PERFORM public.research_lab_routing_assert_provider_receipt_chain_v3(
            attempt.experiment_hash, attempt.binding_id, attempt.tool_id,
            attempt.variant_id, attempt.unit_ref, attempt.action_id,
            attempt.authorization_hash, attempt.authorization_request_hash,
            attempt.authorization_proof_hash, attempt.terminal_receipt_hash,
            attempt.protected_release_receipt_hash, attempt.admission_bundle_hash,
            attempt.terminal_request_hash, attempt.terminal_result_hash,
            attempt.terminal_provider_record_hash,
            attempt.terminal_billing_projection_hash, attempt.outcome,
            attempt.credit_microunits, attempt.latency_ms, attempt.billing_state,
            attempt.authoritative_billed_credit_microunits, attempt.attempt_doc
        );
    END LOOP;
END;
$promotion_chain_v3$;

-- The scoring attestation is not the durable database authority.  Promotion
-- therefore rebuilds the exact row projections used by the Python
-- reconciler, and binds the signed attestation output to those projections.
-- This closes the gap where a caller could present a previously valid
-- attestation while substituting a durable decision, provider, or budget row
-- before the Lab reference was written.
CREATE OR REPLACE FUNCTION public.research_lab_routing_assert_promotion_reconciliation_v3(
    p_experiment_hash TEXT,
    p_evaluation_receipt_id TEXT,
    p_evaluation_hash TEXT,
    p_selected_variant_id TEXT,
    p_reconciliation_doc JSONB
)
RETURNS VOID
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $promotion_reconciliation_v3$
DECLARE
    experiment public.research_lab_routing_experiments_v2%ROWTYPE;
    evaluation public.research_lab_routing_evaluation_receipts_v2%ROWTYPE;
    authority_receipt public.research_lab_attested_execution_receipts_v2%ROWTYPE;
    decision_root TEXT;
    provider_root TEXT;
    budget_root TEXT;
    envelope_hash TEXT;
    billed_total BIGINT;
    expected_output JSONB;
    expected_output_root TEXT;
BEGIN
    IF p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_evaluation_receipt_id !~ '^routing_evaluation_v2:[0-9a-f]{16}$'
       OR p_evaluation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_selected_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR pg_catalog.jsonb_typeof(p_reconciliation_doc) IS DISTINCT FROM 'object'
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_reconciliation_invalid'
            USING ERRCODE = '22023';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );

    SELECT * INTO experiment
      FROM public.research_lab_routing_experiments_v2
     WHERE experiment_hash = p_experiment_hash;
    IF NOT FOUND
       OR public.research_lab_routing_jsonb_hash_v2(experiment.spec_doc)
            IS DISTINCT FROM p_experiment_hash
       OR experiment.execution_envelope_hash IS NULL
       OR experiment.execution_envelope_doc IS NULL
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_experiment_authority_mismatch'
            USING ERRCODE = '23503';
    END IF;

    -- The envelope is the only durable copy of the signed artifact, unit,
    -- gold-label, and model-binding identities.  Recompute its hash and bind
    -- every identity named by the reconciliation document to that copy.
    envelope_hash := public.research_lab_routing_jsonb_hash_v2(
        experiment.execution_envelope_doc
    );
    IF envelope_hash IS DISTINCT FROM experiment.execution_envelope_hash
       OR experiment.execution_envelope_doc->>'schema_version'
            IS DISTINCT FROM 'leadpoet.research_lab.routing_execution_envelope.v2'
       OR experiment.execution_envelope_doc->>'experiment_hash'
            IS DISTINCT FROM p_experiment_hash
       OR experiment.execution_envelope_doc->>'artifact_lineage_hash'
            IS DISTINCT FROM p_reconciliation_doc->>'artifact_lineage_hash'
       OR experiment.execution_envelope_doc->>'pointer_document_hash'
            IS DISTINCT FROM p_reconciliation_doc->>'artifact_pointer_document_hash'
       OR experiment.execution_envelope_doc->>'gold_label_manifest_hash'
            IS DISTINCT FROM p_reconciliation_doc->>'gold_label_manifest_hash'
       OR p_reconciliation_doc->>'execution_envelope_hash'
            IS DISTINCT FROM envelope_hash
       OR p_reconciliation_doc->>'artifact_lineage_hash'
            !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'artifact_pointer_document_hash'
            !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'gold_label_manifest_hash'
            !~ '^sha256:[0-9a-f]{64}$'
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_envelope_authority_mismatch'
            USING ERRCODE = '23503';
    END IF;

    SELECT * INTO evaluation
      FROM public.research_lab_routing_evaluation_receipts_v2
     WHERE receipt_id = p_evaluation_receipt_id;
    IF NOT FOUND
       OR evaluation.experiment_hash IS DISTINCT FROM p_experiment_hash
       OR evaluation.evaluation_hash IS DISTINCT FROM p_evaluation_hash
       OR evaluation.evaluation_hash IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(evaluation.evaluation_doc)
       OR evaluation.selected_variant_id IS DISTINCT FROM p_selected_variant_id
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_evaluation_authority_mismatch'
            USING ERRCODE = '23503';
    END IF;

    -- These are byte-for-byte the projections in
    -- routing_experiment_reconciliation._root().  Do not add created_at or
    -- receipt-only columns here: they are deliberately outside the Python
    -- authority roots.
    SELECT public.research_lab_routing_jsonb_hash_v2(
        coalesce(
            pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'key', decision.receipt_id,
                    'row', pg_catalog.jsonb_build_object(
                        'receipt_id', decision.receipt_id,
                        'experiment_hash', decision.experiment_hash,
                        'decision_doc', decision.decision_doc
                    )
                ) ORDER BY decision.receipt_id
            ), '[]'::JSONB
        )
    ) INTO decision_root
    FROM public.research_lab_routing_decision_receipts_v2 decision
    WHERE decision.experiment_hash = p_experiment_hash;

    SELECT public.research_lab_routing_jsonb_hash_v2(
        coalesce(
            pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'key', attempt.attempt_key,
                    'row', pg_catalog.jsonb_build_object(
                        'attempt_key', attempt.attempt_key,
                        'experiment_hash', attempt.experiment_hash,
                        'provider_receipt_ref', attempt.provider_receipt_ref,
                        'binding_id', attempt.binding_id,
                        'tool_id', attempt.tool_id,
                        'variant_id', attempt.variant_id,
                        'unit_ref', attempt.unit_ref,
                        'reservation_id', attempt.reservation_id,
                        'action_id', attempt.action_id,
                        'binding_catalog_manifest_hash', attempt.binding_catalog_manifest_hash,
                        'authorization_hash', attempt.authorization_hash,
                        'authorization_proof_hash', attempt.authorization_proof_hash,
                        'request_body_hash', attempt.request_body_hash,
                        'request_fingerprint', attempt.request_fingerprint,
                        'outcome', attempt.outcome,
                        'credit_microunits', attempt.credit_microunits,
                        'latency_ms', attempt.latency_ms,
                        'execution_mode', attempt.execution_mode,
                        'billing_state', attempt.billing_state,
                        'authoritative_billed_credit_microunits',
                            attempt.authoritative_billed_credit_microunits,
                        'attempt_doc', attempt.attempt_doc
                    )
                ) ORDER BY attempt.attempt_key
            ), '[]'::JSONB
        )
    ) INTO provider_root
    FROM public.research_lab_routing_provider_attempts_v2 attempt
    WHERE attempt.experiment_hash = p_experiment_hash;

    SELECT public.research_lab_routing_jsonb_hash_v2(
        coalesce(
            pg_catalog.jsonb_agg(
                pg_catalog.jsonb_build_object(
                    'key', budget.event_key,
                    'row', pg_catalog.jsonb_build_object(
                        'event_key', budget.event_key,
                        'experiment_hash', budget.experiment_hash,
                        'reservation_id', budget.reservation_id,
                        'binding_id', budget.binding_id,
                        'attempt_key', budget.attempt_key,
                        'event_type', budget.event_type,
                        'credit_microunits', budget.credit_microunits,
                        'event_doc', budget.event_doc
                    )
                ) ORDER BY budget.event_key
            ), '[]'::JSONB
        )
    ) INTO budget_root
    FROM public.research_lab_routing_budget_events_v2 budget
    WHERE budget.experiment_hash = p_experiment_hash;

    SELECT coalesce(sum(attempt.authoritative_billed_credit_microunits), 0)::BIGINT
      INTO billed_total
      FROM public.research_lab_routing_provider_attempts_v2 attempt
     WHERE attempt.experiment_hash = p_experiment_hash;

    IF p_reconciliation_doc->>'decision_receipts_root' IS DISTINCT FROM decision_root
       OR p_reconciliation_doc->>'provider_attempts_root' IS DISTINCT FROM provider_root
       OR p_reconciliation_doc->>'budget_events_root' IS DISTINCT FROM budget_root
       OR p_reconciliation_doc->>'authority_input_root'
            !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'authoritative_billed_credit_microunits'
            !~ '^[0-9]+$'
       OR (
            CASE
                WHEN (
                    p_reconciliation_doc
                        ->>'authoritative_billed_credit_microunits'
                ) ~ '^[0-9]+$'
                THEN (
                    p_reconciliation_doc
                        ->>'authoritative_billed_credit_microunits'
                )::NUMERIC
                ELSE -1::NUMERIC
            END
          ) IS DISTINCT FROM billed_total::NUMERIC
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_durable_root_mismatch'
            USING ERRCODE = '23503';
    END IF;

    -- The signed scoring result is exactly the object returned by
    -- routing_experiment_reconciliation.receipt_output().  Its output root
    -- is recomputed from durable rows, not accepted from the caller.
    expected_output := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.research_lab.routing_experiment_attestation_result.v2',
        'reconciled', TRUE,
        'experiment_hash', p_experiment_hash,
        'evaluation_hash', p_evaluation_hash,
        'evaluation_receipt_id', p_evaluation_receipt_id,
        'selected_variant_id', p_selected_variant_id,
        'decision_receipts_root', decision_root,
        'provider_attempts_root', provider_root,
        'budget_events_root', budget_root,
        'artifact_lineage_hash', p_reconciliation_doc->>'artifact_lineage_hash',
        'gold_label_manifest_hash', p_reconciliation_doc->>'gold_label_manifest_hash',
        'execution_envelope_hash', envelope_hash,
        'authoritative_billed_credit_microunits', billed_total,
        'input_root', p_reconciliation_doc->>'authority_input_root'
    );
    expected_output_root := public.research_lab_routing_jsonb_hash_v2(expected_output);
    IF p_reconciliation_doc->>'authority_output_root' IS DISTINCT FROM expected_output_root
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_attestation_output_mismatch'
            USING ERRCODE = '23503';
    END IF;

    -- Preserve signed input ancestry.  The full attestation input contains
    -- private signed artifact and label documents and is intentionally not
    -- copied into this database.  The persisted signed receipt is therefore
    -- the authoritative input-root commitment; its self-reference, signer,
    -- status, and output are all checked here.
    SELECT receipt.* INTO authority_receipt
      FROM public.research_lab_attested_execution_receipts_v2 receipt
      JOIN public.research_lab_attested_boot_identities_v2 signer
        ON signer.boot_identity_hash = receipt.boot_identity_hash
       AND signer.signing_pubkey = receipt.enclave_pubkey
     WHERE receipt.receipt_hash = p_reconciliation_doc->>'authority_receipt_hash'
       AND receipt.schema_version = 'leadpoet.attested_execution_receipt.v2'
       AND receipt.role = 'gateway_scoring'
       AND receipt.purpose = 'research_lab.routing_experiment.v2'
       AND receipt.receipt_status = 'succeeded'
       AND receipt.input_root = p_reconciliation_doc->>'authority_input_root'
       AND receipt.output_root = expected_output_root
       AND receipt.receipt_doc->>'receipt_hash' = receipt.receipt_hash
       AND receipt.receipt_doc->>'input_root' = receipt.input_root
       AND receipt.receipt_doc->>'output_root' = receipt.output_root;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_attestation_ancestry_mismatch'
            USING ERRCODE = '23503';
    END IF;
END;
$promotion_reconciliation_v3$;

CREATE OR REPLACE FUNCTION public.research_lab_routing_promote_v3(
    p_reference_hash TEXT,
    p_experiment_hash TEXT,
    p_evaluation_receipt_id TEXT,
    p_evaluation_hash TEXT,
    p_selected_variant_id TEXT,
    p_reconciliation_doc JSONB,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $promote_v3$
DECLARE
    evaluation public.research_lab_routing_evaluation_receipts_v2%ROWTYPE;
    existing public.research_lab_routing_lab_references_v2%ROWTYPE;
BEGIN
    IF p_reference_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_experiment_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_evaluation_receipt_id !~ '^routing_evaluation_v2:[0-9a-f]{16}$'
       OR p_evaluation_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_selected_variant_id !~ '^[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,191}$'
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_reconciliation_doc) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
       OR p_reference_hash IS DISTINCT FROM public.research_lab_routing_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'contract_version', 'leadpoet.routing_experiment_v2_lab_reference:v2',
                'experiment_hash', p_experiment_hash,
                'evaluation_hash', p_evaluation_hash,
                'evaluation_receipt_id', p_evaluation_receipt_id,
                'selected_variant_id', p_selected_variant_id,
                'reconciliation', p_reconciliation_doc
            )
       )
       OR p_event_doc->>'event_type' IS DISTINCT FROM 'promoted'
       OR p_event_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR p_event_doc->>'evaluation_receipt_id'
            IS DISTINCT FROM p_evaluation_receipt_id
       OR p_event_doc->>'reference_hash' IS DISTINCT FROM p_reference_hash
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_invalid_arguments'
            USING ERRCODE = '22023';
    END IF;
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_reconciliation_doc, 'routing reconciliation v3'
    );
    PERFORM public.research_lab_routing_reject_secret_doc_v2(
        p_event_doc, 'routing promotion event v3'
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(p_experiment_hash, 0)
    );
    PERFORM public.research_lab_routing_assert_promotion_reconciliation_v3(
        p_experiment_hash,
        p_evaluation_receipt_id,
        p_evaluation_hash,
        p_selected_variant_id,
        p_reconciliation_doc
    );
    PERFORM public.research_lab_routing_assert_promotion_receipt_chain_v3(
        p_experiment_hash
    );
    SELECT * INTO evaluation
      FROM public.research_lab_routing_evaluation_receipts_v2
     WHERE receipt_id = p_evaluation_receipt_id;
    IF NOT FOUND
       OR evaluation.experiment_hash IS DISTINCT FROM p_experiment_hash
       OR evaluation.evaluation_hash IS DISTINCT FROM p_evaluation_hash
       OR evaluation.evaluation_hash IS DISTINCT FROM
            public.research_lab_routing_jsonb_hash_v2(evaluation.evaluation_doc)
       OR evaluation.selected_variant_id IS DISTINCT FROM p_selected_variant_id
       OR p_reconciliation_doc->>'evaluation_receipt_id'
            IS DISTINCT FROM p_evaluation_receipt_id
       OR p_reconciliation_doc->>'evaluation_hash' IS DISTINCT FROM p_evaluation_hash
       OR p_reconciliation_doc->>'experiment_hash' IS DISTINCT FROM p_experiment_hash
       OR p_reconciliation_doc->>'selected_variant_id'
            IS DISTINCT FROM p_selected_variant_id
       OR (p_reconciliation_doc->>'reconciled')::BOOLEAN IS DISTINCT FROM TRUE
       OR p_reconciliation_doc->>'authority_receipt_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'authority_input_root' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reconciliation_doc->>'authority_output_root' !~ '^sha256:[0-9a-f]{64}$'
       OR NOT EXISTS (
           SELECT 1
             FROM public.research_lab_attested_execution_receipts_v2 authority_receipt
             JOIN public.research_lab_attested_boot_identities_v2 signer
               ON signer.boot_identity_hash = authority_receipt.boot_identity_hash
            WHERE authority_receipt.receipt_hash
                    = p_reconciliation_doc->>'authority_receipt_hash'
              AND authority_receipt.schema_version
                    = 'leadpoet.attested_execution_receipt.v2'
              AND authority_receipt.role = 'gateway_scoring'
              AND authority_receipt.purpose = 'research_lab.routing_experiment.v2'
              AND authority_receipt.receipt_status = 'succeeded'
              AND authority_receipt.input_root
                    = p_reconciliation_doc->>'authority_input_root'
              AND authority_receipt.output_root
                    = p_reconciliation_doc->>'authority_output_root'
              AND authority_receipt.commit_sha
                    = p_reconciliation_doc->>'authority_commit_sha'
              AND authority_receipt.pcr0 = p_reconciliation_doc->>'authority_pcr0'
              AND authority_receipt.build_manifest_hash
                    = p_reconciliation_doc->>'authority_build_manifest_hash'
              AND authority_receipt.boot_identity_hash
                    = p_reconciliation_doc->>'authority_boot_identity_hash'
              AND authority_receipt.receipt_doc->>'receipt_hash'
                    = authority_receipt.receipt_hash
              AND authority_receipt.receipt_doc->>'input_root'
                    = authority_receipt.input_root
              AND authority_receipt.receipt_doc->>'output_root'
                    = authority_receipt.output_root
              AND authority_receipt.enclave_pubkey = signer.signing_pubkey
       )
       OR NOT EXISTS (
           SELECT 1
             FROM public.research_lab_routing_experiments_v2 experiment
            WHERE experiment.experiment_hash = p_experiment_hash
              AND experiment.execution_envelope_hash
                    = p_reconciliation_doc->>'execution_envelope_hash'
              AND experiment.execution_envelope_doc->>'pointer_document_hash'
                    = p_reconciliation_doc->>'artifact_pointer_document_hash'
       )
       OR NOT EXISTS (
           SELECT 1
             FROM public.research_lab_routing_experiments_v2 experiment
             JOIN public.research_lab_attested_execution_receipts_v2 observation_receipt
               ON observation_receipt.receipt_hash
                    = experiment.execution_envelope_doc
                        ->>'model_binding_observation_receipt_hash'
              AND observation_receipt.role = 'gateway_scoring'
              AND observation_receipt.purpose
                    = 'research_lab.routing_model_binding_observation.v2'
              AND observation_receipt.receipt_status = 'succeeded'
              AND observation_receipt.input_root
                    = experiment.execution_envelope_doc #>>
                        '{model_binding_observation,result,request_root}'
              AND observation_receipt.receipt_doc
                    = experiment.execution_envelope_doc #>
                        '{model_binding_observation,receipt}'
             JOIN public.research_lab_attested_boot_identities_v2 observation_signer
               ON observation_signer.boot_identity_hash
                    = observation_receipt.boot_identity_hash
              AND observation_receipt.enclave_pubkey
                    = observation_signer.signing_pubkey
            WHERE experiment.experiment_hash = p_experiment_hash
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_evaluation_mismatch'
            USING ERRCODE = '23503';
    END IF;
    IF pg_catalog.jsonb_array_length(
        coalesce(evaluation.evaluation_doc->'decision_receipt_refs', '[]'::JSONB)
    ) = 0
       OR pg_catalog.jsonb_array_length(
        coalesce(evaluation.evaluation_doc->'provider_receipt_refs', '[]'::JSONB)
    ) = 0
       OR NOT EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements(
                 evaluation.evaluation_doc->'variants'
             ) variant(value)
            WHERE variant.value->>'variant_id' = p_selected_variant_id
              AND (variant.value->>'passed')::BOOLEAN IS TRUE
              AND coalesce(
                    (variant.value #>> '{calibration,adapter_failure_count}')::INTEGER,
                    -1
                  ) = 0
              AND coalesce(
                    (variant.value #>> '{holdout,adapter_failure_count}')::INTEGER,
                    -1
                  ) = 0
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements_text(
                 evaluation.evaluation_doc->'decision_receipt_refs'
             ) ref(receipt_id)
            WHERE NOT EXISTS (
                SELECT 1
                  FROM public.research_lab_routing_decision_receipts_v2 decision
                 WHERE decision.experiment_hash = p_experiment_hash
                   AND decision.receipt_id = ref.receipt_id
            )
       )
       OR EXISTS (
           SELECT 1
             FROM pg_catalog.jsonb_array_elements_text(
                 evaluation.evaluation_doc->'provider_receipt_refs'
             ) ref(provider_receipt_ref)
            WHERE NOT EXISTS (
                SELECT 1
                  FROM public.research_lab_routing_provider_attempts_v2 attempt
                 WHERE attempt.experiment_hash = p_experiment_hash
                   AND attempt.provider_receipt_ref = ref.provider_receipt_ref
                   AND attempt.outcome <> 'adapter_failure'
                   AND attempt.billing_state = 'known'
            )
       )
       OR EXISTS (
           (SELECT decision.receipt_id
              FROM public.research_lab_routing_decision_receipts_v2 decision
             WHERE decision.experiment_hash = p_experiment_hash)
           EXCEPT
           (SELECT ref.receipt_id
              FROM pg_catalog.jsonb_array_elements_text(
                  evaluation.evaluation_doc->'decision_receipt_refs'
              ) ref(receipt_id))
       )
       OR EXISTS (
           (SELECT attempt.provider_receipt_ref
              FROM public.research_lab_routing_provider_attempts_v2 attempt
             WHERE attempt.experiment_hash = p_experiment_hash)
           EXCEPT
           (SELECT ref.provider_receipt_ref
              FROM pg_catalog.jsonb_array_elements_text(
                  evaluation.evaluation_doc->'provider_receipt_refs'
              ) ref(provider_receipt_ref))
       )
       OR EXISTS (
           SELECT 1
             FROM public.research_lab_routing_provider_attempts_v2 attempt
             LEFT JOIN public.research_lab_routing_budget_events_v2 reserve_event
               ON reserve_event.experiment_hash = attempt.experiment_hash
              AND reserve_event.reservation_id = attempt.reservation_id
              AND reserve_event.event_type = 'reserve'
             LEFT JOIN LATERAL (
                 SELECT budget_event.event_type, budget_event.event_key,
                        budget_event.attempt_key
                   FROM public.research_lab_routing_budget_events_v2 budget_event
                  WHERE budget_event.reservation_id = attempt.reservation_id
                  ORDER BY budget_event.created_at DESC, budget_event.event_key DESC
                  LIMIT 1
             ) budget_head ON TRUE
            WHERE attempt.experiment_hash = p_experiment_hash
              AND (reserve_event.event_key IS NULL
                OR reserve_event.binding_id IS DISTINCT FROM attempt.binding_id
                OR reserve_event.claim_key IS DISTINCT FROM attempt.claim_key
                OR reserve_event.claim_generation IS DISTINCT FROM attempt.claim_generation
                OR reserve_event.event_doc->>'unit_ref' IS DISTINCT FROM attempt.unit_ref
                OR reserve_event.event_doc->>'variant_id' IS DISTINCT FROM attempt.variant_id
                OR reserve_event.event_doc->>'request_fingerprint'
                    IS DISTINCT FROM attempt.request_fingerprint
                OR reserve_event.event_doc->>'action_id' IS DISTINCT FROM attempt.action_id
                OR budget_head.event_type IS DISTINCT FROM 'settle'
                OR budget_head.attempt_key IS DISTINCT FROM attempt.attempt_key)
       )
       OR EXISTS (
           SELECT 1
             FROM (
                 SELECT DISTINCT ON (budget_event.reservation_id)
                     budget_event.event_type
                   FROM public.research_lab_routing_budget_events_v2 budget_event
                  WHERE budget_event.experiment_hash = p_experiment_hash
                  ORDER BY budget_event.reservation_id,
                      budget_event.created_at DESC, budget_event.event_key DESC
             ) budget_head
            WHERE budget_head.event_type <> 'settle'
       )
    THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_reconciliation_failed'
            USING ERRCODE = '23503';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.event_hash = p_event_hash
           AND (event.experiment_hash IS DISTINCT FROM p_experiment_hash
             OR event.event_type IS DISTINCT FROM 'promoted'
             OR event.event_doc IS DISTINCT FROM p_event_doc)
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_event_conflict'
            USING ERRCODE = '23505';
    END IF;
    SELECT * INTO existing
      FROM public.research_lab_routing_lab_references_v2
     WHERE reference_hash = p_reference_hash;
    IF FOUND THEN
        IF existing.experiment_hash IS DISTINCT FROM p_experiment_hash
           OR existing.evaluation_receipt_id IS DISTINCT FROM p_evaluation_receipt_id
           OR existing.selected_variant_id IS DISTINCT FROM p_selected_variant_id
           OR existing.reconciliation_doc IS DISTINCT FROM p_reconciliation_doc
        THEN
            RAISE EXCEPTION 'research_lab_routing_promote_v3_conflict'
                USING ERRCODE = '23505';
        END IF;
        IF NOT EXISTS (
            SELECT 1
              FROM public.research_lab_routing_experiment_events_v2 event
             WHERE event.event_hash = p_event_hash
               AND event.experiment_hash = p_experiment_hash
               AND event.event_type = 'promoted'
               AND event.event_doc = p_event_doc
        ) THEN
            RAISE EXCEPTION 'research_lab_routing_promote_v3_event_missing'
                USING ERRCODE = '23503';
        END IF;
        RETURN pg_catalog.jsonb_build_object(
            'reference_hash', p_reference_hash, 'idempotent', TRUE
        );
    END IF;
    INSERT INTO public.research_lab_routing_lab_references_v2 (
        reference_hash, experiment_hash, evaluation_receipt_id,
        selected_variant_id, reconciliation_doc
    ) VALUES (
        p_reference_hash, p_experiment_hash, p_evaluation_receipt_id,
        p_selected_variant_id, p_reconciliation_doc
    );
    INSERT INTO public.research_lab_routing_experiment_events_v2 (
        event_hash, experiment_hash, event_type, event_doc
    ) VALUES (
        p_event_hash, p_experiment_hash, 'promoted', p_event_doc
    ) ON CONFLICT (event_hash) DO NOTHING;
    IF NOT EXISTS (
        SELECT 1
          FROM public.research_lab_routing_experiment_events_v2 event
         WHERE event.event_hash = p_event_hash
           AND event.experiment_hash = p_experiment_hash
           AND event.event_type = 'promoted'
           AND event.event_doc = p_event_doc
    ) THEN
        RAISE EXCEPTION 'research_lab_routing_promote_v3_event_conflict'
            USING ERRCODE = '23505';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'reference_hash', p_reference_hash, 'idempotent', FALSE
    );
END;
$promote_v3$;

ALTER TABLE public.research_lab_routing_experiments_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_events_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_execution_requests_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_claims_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_claim_heartbeats_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_claim_closures_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_claims_v3 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_claim_heartbeats_v3 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_experiment_claim_closures_v3 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_provider_attempts_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_decision_receipts_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_evaluation_receipts_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_lab_references_v2 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_routing_budget_events_v2 ENABLE ROW LEVEL SECURITY;

-- The service role can read the append-only audit trail but cannot mutate it
-- directly. All writes remain SECURITY DEFINER RPC-only.
DROP POLICY IF EXISTS rl_route_experiment_service_read ON public.research_lab_routing_experiments_v2;
CREATE POLICY rl_route_experiment_service_read ON public.research_lab_routing_experiments_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_event_service_read ON public.research_lab_routing_experiment_events_v2;
CREATE POLICY rl_route_event_service_read ON public.research_lab_routing_experiment_events_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_request_service_read
    ON public.research_lab_routing_execution_requests_v2;
CREATE POLICY rl_route_request_service_read
    ON public.research_lab_routing_execution_requests_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_claim_service_read ON public.research_lab_routing_experiment_claims_v2;
CREATE POLICY rl_route_claim_service_read ON public.research_lab_routing_experiment_claims_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_claim_heartbeat_service_read
    ON public.research_lab_routing_experiment_claim_heartbeats_v2;
CREATE POLICY rl_route_claim_heartbeat_service_read
    ON public.research_lab_routing_experiment_claim_heartbeats_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_claim_closure_service_read
    ON public.research_lab_routing_experiment_claim_closures_v2;
CREATE POLICY rl_route_claim_closure_service_read
    ON public.research_lab_routing_experiment_claim_closures_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_claim_v3_service_read
    ON public.research_lab_routing_experiment_claims_v3;
CREATE POLICY rl_route_claim_v3_service_read
    ON public.research_lab_routing_experiment_claims_v3
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_claim_heartbeat_v3_service_read
    ON public.research_lab_routing_experiment_claim_heartbeats_v3;
CREATE POLICY rl_route_claim_heartbeat_v3_service_read
    ON public.research_lab_routing_experiment_claim_heartbeats_v3
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_claim_closure_v3_service_read
    ON public.research_lab_routing_experiment_claim_closures_v3;
CREATE POLICY rl_route_claim_closure_v3_service_read
    ON public.research_lab_routing_experiment_claim_closures_v3
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_attempt_service_read ON public.research_lab_routing_provider_attempts_v2;
CREATE POLICY rl_route_attempt_service_read ON public.research_lab_routing_provider_attempts_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_decision_service_read ON public.research_lab_routing_decision_receipts_v2;
CREATE POLICY rl_route_decision_service_read ON public.research_lab_routing_decision_receipts_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_evaluation_service_read ON public.research_lab_routing_evaluation_receipts_v2;
CREATE POLICY rl_route_evaluation_service_read ON public.research_lab_routing_evaluation_receipts_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_reference_service_read ON public.research_lab_routing_lab_references_v2;
CREATE POLICY rl_route_reference_service_read ON public.research_lab_routing_lab_references_v2
    FOR SELECT TO service_role USING (TRUE);
DROP POLICY IF EXISTS rl_route_budget_service_read ON public.research_lab_routing_budget_events_v2;
CREATE POLICY rl_route_budget_service_read ON public.research_lab_routing_budget_events_v2
    FOR SELECT TO service_role USING (TRUE);

REVOKE ALL ON TABLE public.research_lab_routing_experiments_v2,
    public.research_lab_routing_experiment_events_v2,
    public.research_lab_routing_execution_requests_v2,
    public.research_lab_routing_experiment_claims_v2,
    public.research_lab_routing_experiment_claim_heartbeats_v2,
    public.research_lab_routing_experiment_claim_closures_v2,
    public.research_lab_routing_experiment_claims_v3,
    public.research_lab_routing_experiment_claim_heartbeats_v3,
    public.research_lab_routing_experiment_claim_closures_v3,
    public.research_lab_routing_provider_attempts_v2,
    public.research_lab_routing_decision_receipts_v2,
    public.research_lab_routing_evaluation_receipts_v2,
    public.research_lab_routing_lab_references_v2,
    public.research_lab_routing_budget_events_v2
FROM PUBLIC, anon, authenticated, service_role;
REVOKE TRUNCATE ON TABLE public.research_lab_routing_experiments_v2,
    public.research_lab_routing_experiment_events_v2,
    public.research_lab_routing_execution_requests_v2,
    public.research_lab_routing_experiment_claims_v2,
    public.research_lab_routing_experiment_claim_heartbeats_v2,
    public.research_lab_routing_experiment_claim_closures_v2,
    public.research_lab_routing_experiment_claims_v3,
    public.research_lab_routing_experiment_claim_heartbeats_v3,
    public.research_lab_routing_experiment_claim_closures_v3,
    public.research_lab_routing_provider_attempts_v2,
    public.research_lab_routing_decision_receipts_v2,
    public.research_lab_routing_evaluation_receipts_v2,
    public.research_lab_routing_lab_references_v2,
    public.research_lab_routing_budget_events_v2
FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.research_lab_routing_experiments_v2,
    public.research_lab_routing_experiment_events_v2,
    public.research_lab_routing_execution_requests_v2,
    public.research_lab_routing_experiment_claims_v2,
    public.research_lab_routing_experiment_claim_heartbeats_v2,
    public.research_lab_routing_experiment_claim_closures_v2,
    public.research_lab_routing_experiment_claims_v3,
    public.research_lab_routing_experiment_claim_heartbeats_v3,
    public.research_lab_routing_experiment_claim_closures_v3,
    public.research_lab_routing_provider_attempts_v2,
    public.research_lab_routing_decision_receipts_v2,
    public.research_lab_routing_evaluation_receipts_v2,
    public.research_lab_routing_lab_references_v2,
    public.research_lab_routing_budget_events_v2
TO service_role;

REVOKE ALL ON FUNCTION public.research_lab_routing_append_only_v2() FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_canonical_jsonb_v2(JSONB) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_jsonb_hash_v2(JSONB) FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_reject_secret_doc_v2(JSONB, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_capability_commitment_v2(TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_submit_experiment_v2(TEXT, TEXT, JSONB, TEXT, BOOLEAN, TEXT, JSONB, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_request_execution_v2(TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_claim_v2(TEXT, TEXT, BIGINT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_experiment_v2(TEXT, TEXT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_renew_claim_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_close_claim_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_event_v2(TEXT, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_event_v2(TEXT, TEXT, TEXT, JSONB) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_fenced_event_v2(TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_recover_claim_v2(TEXT, TEXT, TEXT, JSONB, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_provider_attempt_v2(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, BIGINT, BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_decision_receipt_v2(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_evaluation_v2(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_reserve_budget_v2(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, BIGINT, INTEGER, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_settle_budget_v2(TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_mark_budget_uncertain_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_recover_budget_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_list_expired_budget_reservations_v2(TEXT, TEXT, BIGINT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_list_unresolved_budget_reservations_v2(TEXT, TEXT, BIGINT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_routing_promote_v2(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, JSONB) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.research_lab_routing_submit_experiment_v2(TEXT, TEXT, JSONB, TEXT, BOOLEAN, TEXT, JSONB, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_request_execution_v2(TEXT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_claim_experiment_v2(TEXT, TEXT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_renew_claim_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_close_claim_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_fenced_event_v2(TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_recover_claim_v2(TEXT, TEXT, TEXT, JSONB, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_provider_attempt_v2(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, BIGINT, BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_decision_receipt_v2(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_evaluation_v2(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_reserve_budget_v2(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, BIGINT, INTEGER, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_settle_budget_v2(TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_mark_budget_uncertain_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_recover_budget_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_list_expired_budget_reservations_v2(TEXT, TEXT, BIGINT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_list_unresolved_budget_reservations_v2(TEXT, TEXT, BIGINT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_promote_v2(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, JSONB) TO service_role;

REVOKE ALL ON FUNCTION public.research_lab_routing_assert_claim_v3(TEXT, TEXT, BIGINT)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_assert_claim_v3(TEXT, TEXT, BIGINT)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_experiment_v3(TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_claim_experiment_v3(TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_recover_claim_v3(TEXT, TEXT, TEXT, JSONB, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_recover_claim_v3(TEXT, TEXT, TEXT, JSONB, TEXT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_renew_claim_v3(TEXT, TEXT, TEXT, BIGINT, INTEGER, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_renew_claim_v3(TEXT, TEXT, TEXT, BIGINT, INTEGER, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_close_claim_v3(TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_close_claim_v3(TEXT, TEXT, TEXT, BIGINT, TEXT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_fenced_event_v3(TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_fenced_event_v3(TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v3(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, BIGINT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_assert_provider_receipt_chain_v3(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, BIGINT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_provider_attempt_v3(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, BIGINT, BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_provider_attempt_v3(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, BIGINT, BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_decision_receipt_v3(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_decision_receipt_v3(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_append_evaluation_v3(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_append_evaluation_v3(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_reserve_budget_v3(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, BIGINT, INTEGER, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_reserve_budget_v3(TEXT, TEXT, TEXT, TEXT, TEXT, BIGINT, BIGINT, INTEGER, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_settle_budget_v3(TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_settle_budget_v3(TEXT, TEXT, TEXT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_mark_budget_uncertain_v3(TEXT, TEXT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_mark_budget_uncertain_v3(TEXT, TEXT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_recover_budget_v3(TEXT, TEXT, TEXT, BIGINT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_recover_budget_v3(TEXT, TEXT, TEXT, BIGINT, JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_list_expired_budget_reservations_v3(TEXT, TEXT, BIGINT)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_list_expired_budget_reservations_v3(TEXT, TEXT, BIGINT)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_list_unresolved_budget_reservations_v3(TEXT, TEXT, BIGINT)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_list_unresolved_budget_reservations_v3(TEXT, TEXT, BIGINT)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_promotion_receipt_chain_v3(TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_assert_promotion_receipt_chain_v3(TEXT)
    TO service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_promotion_reconciliation_v3(TEXT, TEXT, TEXT, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_promote_v3(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_routing_promote_v3(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, JSONB)
    TO service_role;

-- Retire bearer-capability claim entry points without dropping their
-- overloads.  V3 is the only callable claim authority after this migration.
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_capability_commitment_v2(TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_assert_claim_v2(TEXT, TEXT, BIGINT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_claim_experiment_v2(TEXT, TEXT, TEXT, TEXT, INTEGER, JSONB, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_renew_claim_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, INTEGER, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_routing_close_claim_v2(TEXT, TEXT, TEXT, BIGINT, TEXT, TEXT, JSONB)
    FROM PUBLIC, anon, authenticated, service_role;

COMMENT ON TABLE public.research_lab_routing_experiments_v2 IS
    'Append-only immutable routing experiment specifications; writes only through service-role v2 authority RPCs.';
COMMENT ON TABLE public.research_lab_routing_provider_attempts_v2 IS
    'Redacted provider attempt outcomes only. Adapter failures are retained as evidence but cannot be terminal cache results.';
COMMENT ON TABLE public.research_lab_routing_budget_events_v2 IS
    'Append-only atomic routing provider budget reserve, settle, uncertain, and recovery events.';
COMMENT ON FUNCTION public.research_lab_routing_promote_v2(TEXT, TEXT, TEXT, TEXT, TEXT, JSONB, TEXT, JSONB) IS
    'Creates an immutable Lab reference only after authoritative evaluation and receipt reconciliation; never promotes production routing.';

NOTIFY pgrst, 'reload schema';

COMMIT;
