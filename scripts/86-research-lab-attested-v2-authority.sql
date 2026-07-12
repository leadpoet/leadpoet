-- Research Lab authoritative V2 attestation records.
--
-- Additive only. Migration 80 remains immutable V1 history. These tables have
-- no shadow/optional mode and do not modify existing scoring, benchmark,
-- promotion, reward, allocation, SOURCE_ADD, fulfillment, or weight tables.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_attested_v2_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only; insert a new V2 record', TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_attested_v2_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_attested_v2_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_provider_credential_envelopes_v2 (
    envelope_hash            TEXT        PRIMARY KEY CHECK (envelope_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.provider_credential_envelope.v2'),
    key_ref                  TEXT        NOT NULL CHECK (key_ref ~ '^encrypted_ref:openrouter:[0-9a-f]{32}$'),
    key_ref_hash             TEXT        NOT NULL CHECK (key_ref_hash ~ '^sha256:[0-9a-f]{64}$'),
    miner_hotkey_hash        TEXT        NOT NULL CHECK (miner_hotkey_hash ~ '^sha256:[0-9a-f]{64}$'),
    credential_kind         TEXT        NOT NULL CHECK (credential_kind IN ('runtime', 'management')),
    credential_slot         TEXT        NOT NULL CHECK (
                                            credential_slot IN ('openrouter', 'openrouter_management')
                                          ),
    credential_value_hash   TEXT        NOT NULL CHECK (credential_value_hash ~ '^sha256:[0-9a-f]{64}$'),
    ciphertext_blob_b64     TEXT        NOT NULL CHECK (ciphertext_blob_b64 !~* '(sk-or-|raw_secret|service_role)'),
    ciphertext_blob_hash    TEXT        NOT NULL CHECK (ciphertext_blob_hash ~ '^sha256:[0-9a-f]{64}$'),
    kms_key_id_hash         TEXT        NOT NULL CHECK (kms_key_id_hash ~ '^sha256:[0-9a-f]{64}$'),
    encryption_context      JSONB       NOT NULL CHECK (
                                            jsonb_typeof(encryption_context) = 'object'
                                            AND encryption_context::TEXT !~* '(sk-or-|raw_secret|service_role)'
                                        ),
    encryption_context_hash TEXT        NOT NULL CHECK (encryption_context_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (key_ref, credential_kind)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_boot_identities_v2 (
    boot_identity_hash       TEXT        PRIMARY KEY CHECK (boot_identity_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.attested_boot_identity.v2'),
    role                     TEXT        NOT NULL CHECK (role IN (
                                                'gateway_coordinator',
                                                'gateway_scoring',
                                                'gateway_autoresearch',
                                                'validator_weights'
                                            )),
    physical_role            TEXT        NOT NULL CHECK (physical_role IN (
                                                'gateway_coordinator',
                                                'gateway_scoring_a',
                                                'gateway_scoring_b',
                                                'gateway_autoresearch',
                                                'validator_weights'
                                            )),
    commit_sha               TEXT        NOT NULL CHECK (commit_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    pcr0                     TEXT        NOT NULL CHECK (pcr0 ~ '^[0-9a-f]{96}$'),
    build_manifest_hash      TEXT        NOT NULL CHECK (build_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    dependency_lock_hash     TEXT        NOT NULL CHECK (dependency_lock_hash ~ '^sha256:[0-9a-f]{64}$'),
    config_hash              TEXT        NOT NULL CHECK (config_hash ~ '^sha256:[0-9a-f]{64}$'),
    signing_pubkey           TEXT        NOT NULL CHECK (signing_pubkey ~ '^[0-9a-f]{64}$'),
    transport_pubkey         TEXT        NOT NULL CHECK (transport_pubkey ~ '^[0-9a-f]{64}$'),
    transport_certificate_hash TEXT      NOT NULL CHECK (transport_certificate_hash ~ '^sha256:[0-9a-f]{64}$'),
    boot_nonce               TEXT        NOT NULL CHECK (boot_nonce ~ '^[0-9a-f]{32,64}$'),
    attestation_user_data_hash TEXT      NOT NULL CHECK (attestation_user_data_hash ~ '^sha256:[0-9a-f]{64}$'),
    attestation_document_ref TEXT        NOT NULL,
    attestation_document_hash TEXT       NOT NULL CHECK (attestation_document_hash ~ '^sha256:[0-9a-f]{64}$'),
    identity_doc             JSONB       NOT NULL CHECK (
                                            jsonb_typeof(identity_doc) = 'object'
                                            AND identity_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
                                        ),
    issued_at                TIMESTAMPTZ NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (physical_role, commit_sha, pcr0, boot_nonce)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_transport_attempts_v2 (
    attempt_hash             TEXT        PRIMARY KEY CHECK (attempt_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.attested_transport_attempt.v2'),
    request_id               TEXT        NOT NULL UNIQUE CHECK (request_id ~ '^[0-9a-f]{32}$'),
    logical_operation_id     TEXT        NOT NULL,
    job_id                   TEXT        NOT NULL,
    purpose                  TEXT        NOT NULL CHECK (purpose ~ '\.v2$'),
    provider_id              TEXT        NOT NULL,
    attempt_number           INTEGER     NOT NULL CHECK (attempt_number >= 0),
    request_hash             TEXT        NOT NULL CHECK (request_hash ~ '^sha256:[0-9a-f]{64}$'),
    destination_hash         TEXT        NOT NULL CHECK (destination_hash ~ '^sha256:[0-9a-f]{64}$'),
    terminal_status          TEXT        NOT NULL CHECK (terminal_status IN ('authenticated_response', 'transport_failure')),
    http_status              INTEGER     CHECK (http_status BETWEEN 100 AND 599),
    response_hash            TEXT        CHECK (response_hash IS NULL OR response_hash ~ '^sha256:[0-9a-f]{64}$'),
    request_artifact_hash    TEXT        NOT NULL CHECK (request_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    response_artifact_hash   TEXT        CHECK (response_artifact_hash IS NULL OR response_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    tls_peer_chain_hash      TEXT        CHECK (tls_peer_chain_hash IS NULL OR tls_peer_chain_hash ~ '^sha256:[0-9a-f]{64}$'),
    failure_code             TEXT,
    attempt_doc              JSONB       NOT NULL CHECK (
                                            jsonb_typeof(attempt_doc) = 'object'
                                            AND attempt_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    started_at               TIMESTAMPTZ NOT NULL,
    completed_at             TIMESTAMPTZ NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (completed_at >= started_at),
    CHECK (
        (terminal_status = 'authenticated_response'
            AND http_status IS NOT NULL
            AND response_hash IS NOT NULL
            AND response_artifact_hash IS NOT NULL
            AND tls_peer_chain_hash IS NOT NULL
            AND failure_code IS NULL)
        OR
        (terminal_status = 'transport_failure'
            AND http_status IS NULL
            AND response_hash IS NULL
            AND response_artifact_hash IS NULL
            AND failure_code IS NOT NULL)
    ),
    UNIQUE (logical_operation_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_execution_receipts_v2 (
    receipt_hash             TEXT        PRIMARY KEY CHECK (receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.attested_execution_receipt.v2'),
    role                     TEXT        NOT NULL CHECK (role IN (
                                                'gateway_coordinator',
                                                'gateway_scoring',
                                                'gateway_autoresearch',
                                                'validator_weights'
                                            )),
    purpose                  TEXT        NOT NULL,
    job_id                   TEXT        NOT NULL,
    epoch_id                 INTEGER     NOT NULL CHECK (epoch_id >= 0),
    sequence                 INTEGER     NOT NULL CHECK (sequence >= 0),
    commit_sha               TEXT        NOT NULL CHECK (commit_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    pcr0                     TEXT        NOT NULL CHECK (pcr0 ~ '^[0-9a-f]{96}$'),
    build_manifest_hash      TEXT        NOT NULL CHECK (build_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    dependency_lock_hash     TEXT        NOT NULL CHECK (dependency_lock_hash ~ '^sha256:[0-9a-f]{64}$'),
    config_hash              TEXT        NOT NULL CHECK (config_hash ~ '^sha256:[0-9a-f]{64}$'),
    boot_identity_hash       TEXT        NOT NULL REFERENCES public.research_lab_attested_boot_identities_v2(boot_identity_hash) ON DELETE RESTRICT,
    input_root               TEXT        NOT NULL CHECK (input_root ~ '^sha256:[0-9a-f]{64}$'),
    output_root              TEXT        NOT NULL CHECK (output_root ~ '^sha256:[0-9a-f]{64}$'),
    transport_root           TEXT        NOT NULL CHECK (transport_root ~ '^sha256:[0-9a-f]{64}$'),
    host_operation_root      TEXT        NOT NULL CHECK (host_operation_root ~ '^sha256:[0-9a-f]{64}$'),
    artifact_root            TEXT        NOT NULL CHECK (artifact_root ~ '^sha256:[0-9a-f]{64}$'),
    receipt_status           TEXT        NOT NULL CHECK (receipt_status IN ('succeeded', 'failed')),
    failure_code             TEXT,
    enclave_pubkey           TEXT        NOT NULL CHECK (enclave_pubkey ~ '^[0-9a-f]{64}$'),
    enclave_signature        TEXT        NOT NULL CHECK (enclave_signature ~ '^[0-9a-f]{128}$'),
    receipt_doc              JSONB       NOT NULL CHECK (
                                            jsonb_typeof(receipt_doc) = 'object'
                                            AND receipt_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    issued_at                TIMESTAMPTZ NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (receipt_status = 'succeeded' AND failure_code IS NULL)
        OR (receipt_status = 'failed' AND failure_code IS NOT NULL)
    ),
    CHECK (
        (role = 'gateway_coordinator' AND purpose IN (
            'research_lab.admission.v2',
            'research_lab.provider_evidence.v2',
            'leadpoet.artifact_persistence.v2',
            'research_lab.ranking.v2',
            'research_lab.promotion_decision.v2',
            'research_lab.reward_decision.v2',
            'research_lab.allocation.v2',
            'research_lab.champion_input.v2',
            'research_lab.reimbursement_input.v2',
            'research_lab.source_add_reward_input.v2',
            'research_lab.sourcing_input.v2',
            'research_lab.fulfillment_input.v2',
            'research_lab.leaderboard_input.v2',
            'research_lab.ban_input.v2',
            'research_lab.anomaly_adjustment_input.v2',
            'gateway.weights.publication.v2'
        )) OR
        (role = 'gateway_scoring' AND purpose IN (
            'research_lab.private_model_run.v2',
            'research_lab.candidate_model_run.v2',
            'research_lab.candidate_test.v2',
            'research_lab.company_score.v2',
            'research_lab.candidate_score.v2',
            'research_lab.baseline_score.v2',
            'research_lab.benchmark.v2',
            'research_lab.rebenchmark.v2',
            'research_lab.confirmation_score.v2',
            'research_lab.source_add_judge.v2',
            'qualification.lead_decision.v2',
            'qualification.email_evidence.v2',
            'qualification.sourcing_epoch.v2'
        )) OR
        (role = 'gateway_autoresearch' AND purpose IN (
            'research_lab.source_inspection.v2',
            'research_lab.research_plan.v2',
            'research_lab.patch_draft.v2',
            'research_lab.patch_validation.v2',
            'research_lab.candidate_test.v2',
            'research_lab.candidate_build.v2',
            'research_lab.candidate_decision.v2',
            'research_lab.stale_parent_repair.v2',
            'research_lab.checkpoint.v2',
            'research_lab.openrouter_guard.v2'
        )) OR
        (role = 'validator_weights' AND purpose IN (
            'validator.weight_snapshot.v2',
            'validator.weights.computed.v2',
            'validator.chain_state.v2',
            'validator.metagraph_state.v2',
            'validator.burn_ownership.v2',
            'validator.feature_flags.v2',
            'validator.constants.v2',
            'validator.hotkey_signature.v2',
            'validator.set_weights_extrinsic.v2',
            'validator.weights.finalized.v2'
        ))
    ),
    UNIQUE (role, purpose, job_id, input_root, config_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_receipt_edges_v2 (
    child_receipt_hash       TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    parent_receipt_hash      TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (child_receipt_hash, parent_receipt_hash),
    CHECK (child_receipt_hash <> parent_receipt_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_receipt_transport_v2 (
    receipt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    attempt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_transport_attempts_v2(attempt_hash) ON DELETE RESTRICT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (receipt_hash, attempt_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_host_operations_v2 (
    request_hash             TEXT        PRIMARY KEY CHECK (request_hash ~ '^sha256:[0-9a-f]{64}$'),
    terminal_hash            TEXT        NOT NULL UNIQUE CHECK (terminal_hash ~ '^sha256:[0-9a-f]{64}$'),
    receipt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    job_id                   TEXT        NOT NULL,
    purpose                  TEXT        NOT NULL CHECK (purpose ~ '\.v2$'),
    operation                TEXT        NOT NULL,
    sequence                 INTEGER     NOT NULL CHECK (sequence >= 0),
    terminal_status          TEXT        NOT NULL CHECK (terminal_status IN ('succeeded', 'failed')),
    failure_code             TEXT,
    request_doc              JSONB       NOT NULL CHECK (
                                            jsonb_typeof(request_doc) = 'object'
                                            AND request_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    terminal_doc             JSONB       NOT NULL CHECK (
                                            jsonb_typeof(terminal_doc) = 'object'
                                            AND terminal_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (terminal_status = 'succeeded' AND failure_code IS NULL)
        OR (terminal_status = 'failed' AND failure_code IS NOT NULL)
    ),
    UNIQUE (receipt_hash, sequence)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_artifact_links_v2 (
    link_id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    receipt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    artifact_kind            TEXT        NOT NULL,
    artifact_ref             TEXT        NOT NULL,
    artifact_hash            TEXT        NOT NULL CHECK (artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    encryption_context_hash  TEXT        NOT NULL CHECK (encryption_context_hash ~ '^sha256:[0-9a-f]{64}$'),
    object_lock_mode         TEXT        NOT NULL CHECK (object_lock_mode = 'COMPLIANCE'),
    retain_until             TIMESTAMPTZ NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (receipt_hash, artifact_kind, artifact_ref, artifact_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_business_artifact_links_v2 (
    receipt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    artifact_kind            TEXT        NOT NULL,
    artifact_ref             TEXT        NOT NULL,
    artifact_hash            TEXT        NOT NULL CHECK (artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (artifact_kind, artifact_ref, artifact_hash),
    UNIQUE (receipt_hash, artifact_kind, artifact_ref, artifact_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_signed_transition_commands_v2 (
    command_hash             TEXT        PRIMARY KEY CHECK (command_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.signed_transition_command.v2'),
    operation                TEXT        NOT NULL,
    target                   TEXT        NOT NULL,
    idempotency_key          TEXT        NOT NULL,
    expected_state_hash      TEXT        NOT NULL CHECK (expected_state_hash ~ '^sha256:[0-9a-f]{64}$'),
    payload_hash             TEXT        NOT NULL CHECK (payload_hash ~ '^sha256:[0-9a-f]{64}$'),
    receipt_hash             TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    enclave_pubkey           TEXT        NOT NULL CHECK (enclave_pubkey ~ '^[0-9a-f]{64}$'),
    enclave_signature        TEXT        NOT NULL CHECK (enclave_signature ~ '^[0-9a-f]{128}$'),
    command_doc              JSONB       NOT NULL CHECK (
                                            jsonb_typeof(command_doc) = 'object'
                                            AND command_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    issued_at                TIMESTAMPTZ NOT NULL,
    expires_at               TIMESTAMPTZ NOT NULL,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (expires_at > issued_at),
    UNIQUE (target, idempotency_key)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_weight_bundles_v2 (
    bundle_hash              TEXT        PRIMARY KEY CHECK (bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version           TEXT        NOT NULL CHECK (schema_version = 'leadpoet.published_weight_bundle.v2'),
    netuid                   INTEGER     NOT NULL CHECK (netuid >= 0),
    epoch_id                 INTEGER     NOT NULL CHECK (epoch_id >= 0),
    block                    BIGINT      NOT NULL CHECK (block >= 0),
    validator_hotkey         TEXT        NOT NULL,
    root_receipt_hash        TEXT        NOT NULL REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    weights_hash             TEXT        NOT NULL CHECK (weights_hash ~ '^[0-9a-f]{64}$'),
    snapshot_hash            TEXT        NOT NULL CHECK (snapshot_hash ~ '^sha256:[0-9a-f]{64}$'),
    bundle_doc               JSONB       NOT NULL CHECK (
                                            jsonb_typeof(bundle_doc) = 'object'
                                            AND bundle_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (netuid, epoch_id, validator_hotkey)
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_publication_events_v2 (
    weight_submission_event_hash TEXT    PRIMARY KEY CHECK (weight_submission_event_hash ~ '^sha256:[0-9a-f]{64}$'),
    bundle_hash              TEXT        NOT NULL UNIQUE REFERENCES public.research_lab_attested_weight_bundles_v2(bundle_hash) ON DELETE RESTRICT,
    publication_receipt_hash TEXT        NOT NULL UNIQUE REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    transparency_event_hash  TEXT        NOT NULL UNIQUE CHECK (transparency_event_hash ~ '^sha256:[0-9a-f]{64}$'),
    durable_readback_hash    TEXT        NOT NULL CHECK (durable_readback_hash ~ '^sha256:[0-9a-f]{64}$'),
    publication_doc          JSONB       NOT NULL CHECK (
                                            jsonb_typeof(publication_doc) = 'object'
                                            AND publication_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.research_lab_attested_weight_finalizations_v2 (
    weight_finalization_event_hash TEXT  PRIMARY KEY CHECK (weight_finalization_event_hash ~ '^sha256:[0-9a-f]{64}$'),
    weight_submission_event_hash TEXT    NOT NULL UNIQUE REFERENCES public.research_lab_attested_publication_events_v2(weight_submission_event_hash) ON DELETE RESTRICT,
    bundle_hash              TEXT        NOT NULL UNIQUE REFERENCES public.research_lab_attested_weight_bundles_v2(bundle_hash) ON DELETE RESTRICT,
    finalization_receipt_hash TEXT       NOT NULL UNIQUE REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash) ON DELETE RESTRICT,
    extrinsic_authorization_hash TEXT    NOT NULL UNIQUE CHECK (extrinsic_authorization_hash ~ '^sha256:[0-9a-f]{64}$'),
    extrinsic_hash           TEXT        NOT NULL UNIQUE CHECK (extrinsic_hash ~ '^0x[0-9a-f]{64}$'),
    finalized_block          BIGINT      NOT NULL CHECK (finalized_block >= 0),
    finalized_block_hash     TEXT        NOT NULL CHECK (finalized_block_hash ~ '^[0-9a-f]{64}$'),
    state_transition_hash    TEXT        NOT NULL CHECK (state_transition_hash ~ '^sha256:[0-9a-f]{64}$'),
    finalization_doc         JSONB       NOT NULL CHECK (
                                            jsonb_typeof(finalization_doc) = 'object'
                                            AND finalization_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|proxy-authorization|://[^/]+:[^/@]+@)'
                                        ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_boot_role
    ON public.research_lab_attested_boot_identities_v2(role, commit_sha, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_transport_job
    ON public.research_lab_attested_transport_attempts_v2(job_id, purpose, attempt_number);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_host_operation_job
    ON public.research_lab_attested_host_operations_v2(job_id, purpose, sequence);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_receipts_epoch
    ON public.research_lab_attested_execution_receipts_v2(epoch_id DESC, role, purpose, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_receipts_job
    ON public.research_lab_attested_execution_receipts_v2(job_id, sequence, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_artifacts_ref
    ON public.research_lab_attested_artifact_links_v2(artifact_kind, artifact_ref, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_business_artifacts_receipt
    ON public.research_lab_attested_business_artifact_links_v2(receipt_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_attested_v2_weights_epoch
    ON public.research_lab_attested_weight_bundles_v2(netuid, epoch_id DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_provider_credentials_v2_ref
    ON public.research_lab_provider_credential_envelopes_v2(key_ref, credential_kind);

DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'research_lab_provider_credential_envelopes_v2',
        'research_lab_attested_boot_identities_v2',
        'research_lab_attested_transport_attempts_v2',
        'research_lab_attested_execution_receipts_v2',
        'research_lab_attested_receipt_edges_v2',
        'research_lab_attested_receipt_transport_v2',
        'research_lab_attested_host_operations_v2',
        'research_lab_attested_artifact_links_v2',
        'research_lab_attested_business_artifact_links_v2',
        'research_lab_signed_transition_commands_v2',
        'research_lab_attested_weight_bundles_v2',
        'research_lab_attested_publication_events_v2',
        'research_lab_attested_weight_finalizations_v2'
    ]
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS prevent_%I_mutation ON public.%I', table_name, table_name);
        EXECUTE format(
            'CREATE TRIGGER prevent_%I_mutation BEFORE UPDATE OR DELETE ON public.%I '
            'FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation()',
            table_name,
            table_name
        );
        EXECUTE format('REVOKE ALL ON TABLE public.%I FROM anon, authenticated', table_name);
        EXECUTE format('GRANT SELECT, INSERT ON TABLE public.%I TO service_role', table_name);
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('DROP POLICY IF EXISTS service_role_read ON public.%I', table_name);
        EXECUTE format(
            'CREATE POLICY service_role_read ON public.%I FOR SELECT TO service_role USING (true)',
            table_name
        );
        EXECUTE format('DROP POLICY IF EXISTS service_role_insert ON public.%I', table_name);
        EXECUTE format(
            'CREATE POLICY service_role_insert ON public.%I FOR INSERT TO service_role WITH CHECK (true)',
            table_name
        );
    END LOOP;
END;
$$;

COMMENT ON TABLE public.research_lab_attested_boot_identities_v2 IS
    'Append-only Nitro boot identities binding role, code, PCR0, dependency closure, and ephemeral signing/transport keys.';
COMMENT ON TABLE public.research_lab_attested_transport_attempts_v2 IS
    'Append-only terminal records for enclave-originated HTTPS attempts. Host/network failures cannot claim provider HTTP status.';
COMMENT ON TABLE public.research_lab_attested_host_operations_v2 IS
    'Append-only signed request/terminal records for enclave-requested host Docker, ECR, filesystem, and database operations.';
COMMENT ON TABLE public.research_lab_attested_execution_receipts_v2 IS
    'Authoritative V2 enclave execution receipts. Existing business rows require complete receipt ancestry before weight authority.';
COMMENT ON TABLE public.research_lab_attested_business_artifact_links_v2 IS
    'Append-only unambiguous links from existing content-addressed business artifacts to their authoritative V2 receipt roots.';
COMMENT ON TABLE public.research_lab_attested_weight_bundles_v2 IS
    'Authoritative V2 enclave-computed weight bundles. No shadow or optional persistence mode.';
COMMENT ON TABLE public.research_lab_attested_publication_events_v2 IS
    'Durably persisted V2 gateway publication events required before validator extrinsic signing.';
COMMENT ON TABLE public.research_lab_attested_weight_finalizations_v2 IS
    'Append-only validator-enclave proof that the exact authorized weight extrinsic changed finalized chain state.';

COMMIT;

-- Production verification after applying:
-- SELECT table_name
-- FROM information_schema.tables
-- WHERE table_schema = 'public'
--   AND table_name LIKE 'research_lab_attested%_v2'
-- ORDER BY table_name;
--
-- SELECT relname, relrowsecurity
-- FROM pg_class
-- WHERE relnamespace = 'public'::regnamespace
--   AND relname LIKE 'research_lab_attested%_v2'
-- ORDER BY relname;
