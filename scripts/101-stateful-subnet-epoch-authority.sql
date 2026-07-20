-- Official stateful subnet epoch authority and collision-safe settlement bridge.
--
-- Additive only. This migration does not insert or activate a cutover and does
-- not rewrite any legacy epoch_id. It creates an append-only authority for:
--   * one explicit legacy -> stateful cutover per chain genesis/netuid;
--   * one bijective official-epoch -> settlement-epoch boundary mapping; and
--   * exact finalized chain snapshots pinned to one block hash.
--
-- Apply after scripts/99-research-lab-v2-champion-settlement.sql and after the
-- non-transactional concurrent-index prerequisite in
-- scripts/100-stateful-subnet-epoch-high-water-indexes.concurrent.sql. The
-- runtime must remain in legacy_global_360_v1 mode until an operator inserts a
-- fully receipt-backed cutover at an observed finalized subnet epoch boundary.

BEGIN;

-- Installing the write-fence triggers takes brief relation locks. Fail the
-- whole transactional migration instead of waiting indefinitely behind a
-- long-running production transaction; a later retry is safe and additive.
SET LOCAL lock_timeout = '5s';

-- Permit the coordinator enclave to attest the cutover authority document.
-- Migration 99 owns the previously canonical superset, so replace only the
-- role/purpose CHECK and retain every already-deployed purpose.
DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.research_lab_attested_execution_receipts_v2'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%gateway_coordinator%'
          AND pg_get_constraintdef(oid) LIKE '%purpose%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_attested_execution_receipts_v2 DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_attested_execution_receipts_v2
    ADD CONSTRAINT research_lab_attested_execution_receipts_v2_role_purpose_check
    CHECK (
        (role = 'gateway_coordinator' AND purpose IN (
            'research_lab.admission.v2',
            'research_lab.provider_evidence.v2',
            'research_lab.provider_outcome_snapshot.v2',
            'research_lab.provider_outcome_state.v2',
            'research_lab.active_private_model.v2',
            'leadpoet.artifact_persistence.v2',
            'research_lab.ranking.v2',
            'research_lab.promotion_decision.v2',
            'research_lab.reward_decision.v2',
            'research_lab.legacy_finalized_allocation.v2',
            'research_lab.subnet_epoch_cutover.v2',
            'research_lab.source_add_provenance.v2',
            'research_lab.source_add_functional_probe.v2',
            'research_lab.source_add_catalog_snapshot.v2',
            'research_lab.source_add_credential.v2',
            'research_lab.openrouter_credential.v2',
            'research_lab.openrouter_credit_preflight.v2',
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
            'research_lab.provider_evidence_tape.v2',
            'research_lab.candidate_test.v2',
            'research_lab.company_score.v2',
            'research_lab.provider_preflight.v2',
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
            'validator.subnet_epoch_snapshot.v2',
            'validator.chain_state.v2',
            'validator.metagraph_state.v2',
            'validator.burn_ownership.v2',
            'validator.feature_flags.v2',
            'validator.constants.v2',
            'validator.hotkey_signature.v2',
            'validator.serve_axon_extrinsic.v2',
            'validator.set_weights_extrinsic.v2',
            'validator.weights.finalized.v2'
        ))
    ) NOT VALID;
ALTER TABLE public.research_lab_attested_execution_receipts_v2
    VALIDATE CONSTRAINT research_lab_attested_execution_receipts_v2_role_purpose_check;

-- A first stateful boundary must be captured and made durable while the live
-- system is still paused in legacy mode.  This staging ledger breaks the
-- otherwise circular dependency between the validator-signed first boundary
-- and the cutover row that declares its mapping.  Candidate rows do not
-- activate anything and cannot be used as workflow epoch authority.
CREATE TABLE IF NOT EXISTS public.research_lab_stateful_subnet_epoch_candidates_v1 (
    snapshot_hash                         TEXT        PRIMARY KEY
                                                     CHECK (snapshot_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version                        TEXT        NOT NULL
                                                     CHECK (schema_version = 'leadpoet.subnet_epoch_snapshot.v1'),
    mapping_hash                          TEXT        NOT NULL
                                                     CHECK (mapping_hash ~ '^sha256:[0-9a-f]{64}$'),
    epoch_scheme                          TEXT        NOT NULL
                                                     CHECK (epoch_scheme = 'bittensor.subnet_epoch_index.v1'),
    network_genesis_hash                  TEXT        NOT NULL
                                                     CHECK (network_genesis_hash ~ '^0x[0-9a-f]{64}$'),
    netuid                                INTEGER     NOT NULL CHECK (netuid > 0),
    head_kind                             TEXT        NOT NULL CHECK (head_kind = 'finalized'),
    block_hash                            TEXT        NOT NULL
                                                     CHECK (block_hash ~ '^0x[0-9a-f]{64}$'),
    current_block                         BIGINT      NOT NULL CHECK (current_block > 0),
    last_epoch_block                      BIGINT      NOT NULL CHECK (last_epoch_block > 0),
    pending_epoch_at                      BIGINT      NOT NULL CHECK (pending_epoch_at >= 0),
    subnet_epoch_index                    BIGINT      NOT NULL CHECK (subnet_epoch_index >= 0),
    epoch_ref                             TEXT        NOT NULL
                                                     CHECK (epoch_ref ~ '^sha256:[0-9a-f]{64}$'),
    proposed_settlement_epoch_id          INTEGER     NOT NULL CHECK (proposed_settlement_epoch_id >= 0),
    validator_hotkey                      TEXT        NOT NULL
                                                     CHECK (validator_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{40,64}$'),
    candidate_payload_hash                TEXT        NOT NULL
                                                     CHECK (candidate_payload_hash ~ '^sha256:[0-9a-f]{64}$'),
    validator_hotkey_signature            TEXT        NOT NULL
                                                     CHECK (validator_hotkey_signature ~ '^0x[0-9a-f]{128}$'),
    candidate_authorization_hash          TEXT        NOT NULL UNIQUE
                                                     CHECK (candidate_authorization_hash ~ '^sha256:[0-9a-f]{64}$'),
    tempo                                 INTEGER     NOT NULL CHECK (tempo BETWEEN 1 AND 65535),
    blocks_since_last_step                BIGINT      NOT NULL CHECK (blocks_since_last_step >= 0),
    next_epoch_block                      BIGINT      NOT NULL CHECK (next_epoch_block >= 0),
    blocks_remaining                      BIGINT      NOT NULL CHECK (blocks_remaining >= 0),
    chain_state_receipt_hash              TEXT        NOT NULL UNIQUE
                                                     REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                     ON DELETE RESTRICT,
    snapshot_doc                          JSONB       NOT NULL CHECK (
                                                     jsonb_typeof(snapshot_doc) = 'object'
                                                     AND snapshot_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                     ),
    observed_at                           TIMESTAMPTZ NOT NULL,
    created_at                            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (mapping_hash),
    UNIQUE (network_genesis_hash, netuid, subnet_epoch_index, block_hash),
    CHECK (current_block = last_epoch_block),
    CHECK (next_epoch_block = CASE
        WHEN blocks_since_last_step > 50400 THEN current_block
        ELSE LEAST(
            last_epoch_block + tempo,
            CASE
                WHEN pending_epoch_at > 0 THEN pending_epoch_at
                ELSE last_epoch_block + tempo
            END,
            current_block + (50401 - blocks_since_last_step)
        )
    END),
    CHECK (blocks_remaining = GREATEST(0, next_epoch_block - current_block)),
    CHECK (snapshot_doc ?& ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ]),
    CHECK (snapshot_doc - ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ] = '{}'::JSONB),
    CHECK (snapshot_doc->>'schema_version' = schema_version),
    CHECK (snapshot_doc->>'epoch_scheme' = epoch_scheme),
    CHECK (snapshot_doc->>'network_genesis_hash' = network_genesis_hash),
    CHECK ((snapshot_doc->>'netuid')::INTEGER = netuid),
    CHECK (snapshot_doc->>'head_kind' = head_kind),
    CHECK (snapshot_doc->>'block_hash' = block_hash),
    CHECK ((snapshot_doc->>'current_block')::BIGINT = current_block),
    CHECK ((snapshot_doc->>'last_epoch_block')::BIGINT = last_epoch_block),
    CHECK ((snapshot_doc->>'pending_epoch_at')::BIGINT = pending_epoch_at),
    CHECK ((snapshot_doc->>'subnet_epoch_index')::BIGINT = subnet_epoch_index),
    CHECK ((snapshot_doc->>'tempo')::INTEGER = tempo),
    CHECK ((snapshot_doc->>'blocks_since_last_step')::BIGINT = blocks_since_last_step),
    CHECK ((snapshot_doc->>'observed_at')::TIMESTAMPTZ = observed_at),
    CHECK ((snapshot_doc->>'epoch_id')::BIGINT = subnet_epoch_index),
    CHECK (snapshot_doc->>'epoch_ref' = epoch_ref),
    CHECK ((snapshot_doc->>'epoch_block')::BIGINT = 0),
    CHECK ((snapshot_doc->>'next_epoch_block')::BIGINT = next_epoch_block),
    CHECK ((snapshot_doc->>'blocks_remaining')::BIGINT = blocks_remaining),
    CHECK ((snapshot_doc->>'settlement_epoch_id')::INTEGER = proposed_settlement_epoch_id),
    CHECK (snapshot_doc->>'cutover_mapping_hash' = mapping_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_stateful_subnet_epoch_cutovers_v1 (
    cutover_authority_hash                  TEXT        PRIMARY KEY
                                                        CHECK (cutover_authority_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version                          TEXT        NOT NULL
                                                        CHECK (schema_version = 'leadpoet.subnet_epoch_cutover_authority.v1'),
    mapping_hash                            TEXT        NOT NULL UNIQUE
                                                        CHECK (mapping_hash ~ '^sha256:[0-9a-f]{64}$'),
    manifest_schema_version                 TEXT        NOT NULL
                                                        CHECK (manifest_schema_version = 'leadpoet.subnet_epoch_cutover.v1'),
    epoch_scheme                            TEXT        NOT NULL
                                                        CHECK (epoch_scheme = 'bittensor.subnet_epoch_index.v1'),
    previous_epoch_scheme                   TEXT        NOT NULL
                                                        CHECK (previous_epoch_scheme = 'legacy_global_360_v1'),
    network_genesis_hash                    TEXT        NOT NULL
                                                        CHECK (network_genesis_hash ~ '^0x[0-9a-f]{64}$'),
    netuid                                  INTEGER     NOT NULL CHECK (netuid > 0),
    cutover_block                           BIGINT      NOT NULL CHECK (cutover_block >= 0),
    cutover_block_hash                      TEXT        NOT NULL
                                                        CHECK (cutover_block_hash ~ '^0x[0-9a-f]{64}$'),
    first_subnet_epoch_index                BIGINT      NOT NULL CHECK (first_subnet_epoch_index >= 0),
    first_epoch_ref                         TEXT        NOT NULL UNIQUE
                                                        CHECK (first_epoch_ref ~ '^sha256:[0-9a-f]{64}$'),
    first_settlement_epoch_id               INTEGER     NOT NULL CHECK (first_settlement_epoch_id >= 0),
    last_legacy_epoch_id                    INTEGER     NOT NULL CHECK (last_legacy_epoch_id >= 0),
    first_tempo                             INTEGER     NOT NULL CHECK (first_tempo BETWEEN 1 AND 65535),
    first_pending_epoch_at                  BIGINT      NOT NULL CHECK (first_pending_epoch_at >= 0),
    first_blocks_since_last_step            BIGINT      NOT NULL CHECK (first_blocks_since_last_step >= 0),
    first_next_epoch_block                  BIGINT      NOT NULL CHECK (first_next_epoch_block >= 0),
    first_observed_at                       TIMESTAMPTZ NOT NULL,
    first_snapshot_hash                     TEXT        NOT NULL UNIQUE
                                                        REFERENCES public.research_lab_stateful_subnet_epoch_candidates_v1(snapshot_hash)
                                                        ON DELETE RESTRICT,
    first_snapshot_receipt_hash             TEXT        NOT NULL UNIQUE
                                                        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                        ON DELETE RESTRICT,
    last_legacy_bundle_hash                 TEXT        NOT NULL UNIQUE
                                                        REFERENCES public.research_lab_attested_weight_bundles_v2(bundle_hash)
                                                        ON DELETE RESTRICT,
    last_legacy_weight_finalization_event_hash TEXT    NOT NULL UNIQUE
                                                        REFERENCES public.research_lab_attested_weight_finalizations_v2(weight_finalization_event_hash)
                                                        ON DELETE RESTRICT,
    last_legacy_finalization_receipt_hash   TEXT        NOT NULL UNIQUE
                                                        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                        ON DELETE RESTRICT,
    cutover_receipt_hash                    TEXT        NOT NULL UNIQUE
                                                        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                        ON DELETE RESTRICT,
    manifest_doc                            JSONB       NOT NULL CHECK (
                                                        jsonb_typeof(manifest_doc) = 'object'
                                                        AND manifest_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                        ),
    first_snapshot_doc                      JSONB       NOT NULL CHECK (
                                                        jsonb_typeof(first_snapshot_doc) = 'object'
                                                        AND first_snapshot_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                        ),
    authority_doc                           JSONB       NOT NULL CHECK (
                                                        jsonb_typeof(authority_doc) = 'object'
                                                        AND authority_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                        ),
    created_at                              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (network_genesis_hash, netuid),
    UNIQUE (network_genesis_hash, netuid, first_subnet_epoch_index),
    UNIQUE (network_genesis_hash, netuid, first_settlement_epoch_id),
    CHECK (first_settlement_epoch_id::BIGINT = last_legacy_epoch_id::BIGINT + 1),
    CHECK (first_next_epoch_block = LEAST(
        cutover_block + first_tempo,
        CASE
            WHEN first_pending_epoch_at > 0 THEN first_pending_epoch_at
            ELSE cutover_block + first_tempo
        END
    )),
    CHECK (manifest_doc ?& ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'cutover_block', 'cutover_block_hash', 'first_subnet_epoch_index',
        'first_settlement_epoch_id', 'last_legacy_epoch_id', 'mapping_hash'
    ]),
    CHECK (manifest_doc - ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'cutover_block', 'cutover_block_hash', 'first_subnet_epoch_index',
        'first_settlement_epoch_id', 'last_legacy_epoch_id', 'mapping_hash'
    ] = '{}'::JSONB),
    CHECK (manifest_doc->>'schema_version' = manifest_schema_version),
    CHECK (manifest_doc->>'epoch_scheme' = epoch_scheme),
    CHECK (manifest_doc->>'network_genesis_hash' = network_genesis_hash),
    CHECK ((manifest_doc->>'netuid')::INTEGER = netuid),
    CHECK ((manifest_doc->>'cutover_block')::BIGINT = cutover_block),
    CHECK (manifest_doc->>'cutover_block_hash' = cutover_block_hash),
    CHECK ((manifest_doc->>'first_subnet_epoch_index')::BIGINT = first_subnet_epoch_index),
    CHECK ((manifest_doc->>'first_settlement_epoch_id')::INTEGER = first_settlement_epoch_id),
    CHECK ((manifest_doc->>'last_legacy_epoch_id')::INTEGER = last_legacy_epoch_id),
    CHECK (manifest_doc->>'mapping_hash' = mapping_hash),
    CHECK (first_snapshot_doc ?& ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ]),
    CHECK (first_snapshot_doc - ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ] = '{}'::JSONB),
    CHECK (first_snapshot_doc->>'schema_version' = 'leadpoet.subnet_epoch_snapshot.v1'),
    CHECK (first_snapshot_doc->>'epoch_scheme' = epoch_scheme),
    CHECK (first_snapshot_doc->>'network_genesis_hash' = network_genesis_hash),
    CHECK ((first_snapshot_doc->>'netuid')::INTEGER = netuid),
    CHECK (first_snapshot_doc->>'head_kind' = 'finalized'),
    CHECK (first_snapshot_doc->>'block_hash' = cutover_block_hash),
    CHECK ((first_snapshot_doc->>'current_block')::BIGINT = cutover_block),
    CHECK ((first_snapshot_doc->>'last_epoch_block')::BIGINT = cutover_block),
    CHECK ((first_snapshot_doc->>'pending_epoch_at')::BIGINT = first_pending_epoch_at),
    CHECK ((first_snapshot_doc->>'subnet_epoch_index')::BIGINT = first_subnet_epoch_index),
    CHECK ((first_snapshot_doc->>'tempo')::INTEGER = first_tempo),
    CHECK ((first_snapshot_doc->>'blocks_since_last_step')::BIGINT = first_blocks_since_last_step),
    CHECK ((first_snapshot_doc->>'observed_at')::TIMESTAMPTZ = first_observed_at),
    CHECK ((first_snapshot_doc->>'epoch_id')::BIGINT = first_subnet_epoch_index),
    CHECK (first_snapshot_doc->>'epoch_ref' = first_epoch_ref),
    CHECK ((first_snapshot_doc->>'epoch_block')::BIGINT = 0),
    CHECK ((first_snapshot_doc->>'next_epoch_block')::BIGINT = first_next_epoch_block),
    CHECK ((first_snapshot_doc->>'blocks_remaining')::BIGINT = GREATEST(0, first_next_epoch_block - cutover_block)),
    CHECK ((first_snapshot_doc->>'settlement_epoch_id')::INTEGER = first_settlement_epoch_id),
    CHECK (first_snapshot_doc->>'cutover_mapping_hash' = mapping_hash),
    CHECK (authority_doc ?& ARRAY[
        'schema_version', 'mapping_hash', 'first_epoch_ref',
        'first_snapshot_hash', 'first_snapshot_receipt_hash',
        'last_legacy_bundle_hash', 'last_legacy_weight_finalization_event_hash',
        'last_legacy_finalization_receipt_hash', 'manifest'
    ]),
    CHECK (authority_doc - ARRAY[
        'schema_version', 'mapping_hash', 'first_epoch_ref',
        'first_snapshot_hash', 'first_snapshot_receipt_hash',
        'last_legacy_bundle_hash', 'last_legacy_weight_finalization_event_hash',
        'last_legacy_finalization_receipt_hash', 'manifest'
    ] = '{}'::JSONB),
    CHECK (authority_doc->>'schema_version' = schema_version),
    CHECK (authority_doc->>'mapping_hash' = mapping_hash),
    CHECK (authority_doc->>'first_epoch_ref' = first_epoch_ref),
    CHECK (authority_doc->>'first_snapshot_hash' = first_snapshot_hash),
    CHECK (authority_doc->>'first_snapshot_receipt_hash' = first_snapshot_receipt_hash),
    CHECK (authority_doc->>'last_legacy_bundle_hash' = last_legacy_bundle_hash),
    CHECK (authority_doc->>'last_legacy_weight_finalization_event_hash' = last_legacy_weight_finalization_event_hash),
    CHECK (authority_doc->>'last_legacy_finalization_receipt_hash' = last_legacy_finalization_receipt_hash),
    CHECK (authority_doc->'manifest' = manifest_doc)
);

CREATE TABLE IF NOT EXISTS public.research_lab_stateful_subnet_epoch_boundaries_v1 (
    boundary_hash                         TEXT        PRIMARY KEY
                                                      CHECK (boundary_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version                        TEXT        NOT NULL
                                                      CHECK (schema_version = 'leadpoet.subnet_epoch_boundary.v1'),
    mapping_hash                          TEXT        NOT NULL
                                                      REFERENCES public.research_lab_stateful_subnet_epoch_cutovers_v1(mapping_hash)
                                                      ON DELETE RESTRICT,
    epoch_scheme                          TEXT        NOT NULL
                                                      CHECK (epoch_scheme = 'bittensor.subnet_epoch_index.v1'),
    network_genesis_hash                  TEXT        NOT NULL
                                                      CHECK (network_genesis_hash ~ '^0x[0-9a-f]{64}$'),
    netuid                                INTEGER     NOT NULL CHECK (netuid > 0),
    subnet_epoch_index                    BIGINT      NOT NULL CHECK (subnet_epoch_index >= 0),
    epoch_ref                             TEXT        NOT NULL UNIQUE
                                                      CHECK (epoch_ref ~ '^sha256:[0-9a-f]{64}$'),
    settlement_epoch_id                   INTEGER     NOT NULL CHECK (settlement_epoch_id >= 0),
    boundary_block                        BIGINT      NOT NULL CHECK (boundary_block >= 0),
    boundary_block_hash                   TEXT        NOT NULL
                                                      CHECK (boundary_block_hash ~ '^0x[0-9a-f]{64}$'),
    tempo                                 INTEGER     NOT NULL CHECK (tempo BETWEEN 1 AND 65535),
    pending_epoch_at                      BIGINT      NOT NULL CHECK (pending_epoch_at >= 0),
    blocks_since_last_step                BIGINT      NOT NULL CHECK (blocks_since_last_step >= 0),
    next_epoch_block                      BIGINT      NOT NULL CHECK (next_epoch_block >= 0),
    chain_state_receipt_hash              TEXT        NOT NULL UNIQUE
                                                      REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                      ON DELETE RESTRICT,
    boundary_doc                          JSONB       NOT NULL CHECK (
                                                      jsonb_typeof(boundary_doc) = 'object'
                                                      AND boundary_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                      ),
    observed_at                           TIMESTAMPTZ NOT NULL,
    created_at                            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (mapping_hash, subnet_epoch_index),
    UNIQUE (mapping_hash, settlement_epoch_id),
    UNIQUE (mapping_hash, boundary_block),
    UNIQUE (mapping_hash, boundary_block_hash),
    CHECK (next_epoch_block = CASE
        WHEN blocks_since_last_step > 50400 THEN boundary_block
        ELSE LEAST(
            boundary_block + tempo,
            CASE
                WHEN pending_epoch_at > 0 THEN pending_epoch_at
                ELSE boundary_block + tempo
            END,
            boundary_block + (50401 - blocks_since_last_step)
        )
    END),
    CHECK (boundary_doc ?& ARRAY[
        'schema_version', 'mapping_hash', 'snapshot'
    ]),
    CHECK (boundary_doc - ARRAY[
        'schema_version', 'mapping_hash', 'snapshot'
    ] = '{}'::JSONB),
    CHECK (boundary_doc->>'schema_version' = schema_version),
    CHECK (boundary_doc->>'mapping_hash' = mapping_hash),
    CHECK (jsonb_typeof(boundary_doc->'snapshot') = 'object'),
    CHECK ((boundary_doc->'snapshot') ?& ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ]),
    CHECK ((boundary_doc->'snapshot') - ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ] = '{}'::JSONB),
    CHECK ((boundary_doc->'snapshot')->>'schema_version' = 'leadpoet.subnet_epoch_snapshot.v1'),
    CHECK ((boundary_doc->'snapshot')->>'epoch_scheme' = epoch_scheme),
    CHECK ((boundary_doc->'snapshot')->>'network_genesis_hash' = network_genesis_hash),
    CHECK (((boundary_doc->'snapshot')->>'netuid')::INTEGER = netuid),
    CHECK ((boundary_doc->'snapshot')->>'head_kind' = 'finalized'),
    CHECK ((boundary_doc->'snapshot')->>'block_hash' = boundary_block_hash),
    CHECK (((boundary_doc->'snapshot')->>'current_block')::BIGINT = boundary_block),
    CHECK (((boundary_doc->'snapshot')->>'last_epoch_block')::BIGINT = boundary_block),
    CHECK (((boundary_doc->'snapshot')->>'pending_epoch_at')::BIGINT = pending_epoch_at),
    CHECK (((boundary_doc->'snapshot')->>'subnet_epoch_index')::BIGINT = subnet_epoch_index),
    CHECK (((boundary_doc->'snapshot')->>'tempo')::INTEGER = tempo),
    CHECK (((boundary_doc->'snapshot')->>'blocks_since_last_step')::BIGINT = blocks_since_last_step),
    CHECK (((boundary_doc->'snapshot')->>'observed_at')::TIMESTAMPTZ = observed_at),
    CHECK (((boundary_doc->'snapshot')->>'epoch_id')::BIGINT = subnet_epoch_index),
    CHECK ((boundary_doc->'snapshot')->>'epoch_ref' = epoch_ref),
    CHECK (((boundary_doc->'snapshot')->>'epoch_block')::BIGINT = 0),
    CHECK (((boundary_doc->'snapshot')->>'next_epoch_block')::BIGINT = next_epoch_block),
    CHECK (((boundary_doc->'snapshot')->>'blocks_remaining')::BIGINT = GREATEST(0, next_epoch_block - boundary_block)),
    CHECK (((boundary_doc->'snapshot')->>'settlement_epoch_id')::INTEGER = settlement_epoch_id),
    CHECK ((boundary_doc->'snapshot')->>'cutover_mapping_hash' = mapping_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_stateful_subnet_epoch_snapshots_v1 (
    snapshot_hash                         TEXT        PRIMARY KEY
                                                     CHECK (snapshot_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version                        TEXT        NOT NULL
                                                     CHECK (schema_version = 'leadpoet.subnet_epoch_snapshot.v1'),
    mapping_hash                          TEXT        NOT NULL
                                                     REFERENCES public.research_lab_stateful_subnet_epoch_cutovers_v1(mapping_hash)
                                                     ON DELETE RESTRICT,
    epoch_scheme                          TEXT        NOT NULL
                                                     CHECK (epoch_scheme = 'bittensor.subnet_epoch_index.v1'),
    network_genesis_hash                  TEXT        NOT NULL
                                                     CHECK (network_genesis_hash ~ '^0x[0-9a-f]{64}$'),
    netuid                                INTEGER     NOT NULL CHECK (netuid > 0),
    head_kind                             TEXT        NOT NULL CHECK (head_kind IN ('finalized', 'exact')),
    block_hash                            TEXT        NOT NULL
                                                     CHECK (block_hash ~ '^0x[0-9a-f]{64}$'),
    current_block                         BIGINT      NOT NULL CHECK (current_block >= 0),
    last_epoch_block                      BIGINT      NOT NULL CHECK (last_epoch_block >= 0),
    pending_epoch_at                      BIGINT      NOT NULL CHECK (pending_epoch_at >= 0),
    subnet_epoch_index                    BIGINT      NOT NULL CHECK (subnet_epoch_index >= 0),
    epoch_ref                             TEXT        NOT NULL
                                                     CHECK (epoch_ref ~ '^sha256:[0-9a-f]{64}$'),
    settlement_epoch_id                   INTEGER     NOT NULL CHECK (settlement_epoch_id >= 0),
    tempo                                 INTEGER     NOT NULL CHECK (tempo BETWEEN 1 AND 65535),
    blocks_since_last_step                BIGINT      NOT NULL CHECK (blocks_since_last_step >= 0),
    epoch_block                           BIGINT      NOT NULL CHECK (epoch_block >= 0),
    next_epoch_block                      BIGINT      NOT NULL CHECK (next_epoch_block >= 0),
    blocks_remaining                      BIGINT      NOT NULL CHECK (blocks_remaining >= 0),
    chain_state_receipt_hash              TEXT        NOT NULL UNIQUE
                                                     REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                                     ON DELETE RESTRICT,
    snapshot_doc                          JSONB       NOT NULL CHECK (
                                                     jsonb_typeof(snapshot_doc) = 'object'
                                                     AND snapshot_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                                     ),
    observed_at                           TIMESTAMPTZ NOT NULL,
    created_at                            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (mapping_hash, current_block),
    UNIQUE (mapping_hash, block_hash),
    CHECK (current_block >= last_epoch_block),
    CHECK (epoch_block = current_block - last_epoch_block),
    CHECK (next_epoch_block = CASE
        WHEN blocks_since_last_step > 50400 THEN current_block
        ELSE LEAST(
            last_epoch_block + tempo,
            CASE
                WHEN pending_epoch_at > 0 THEN pending_epoch_at
                ELSE last_epoch_block + tempo
            END,
            current_block + (50401 - blocks_since_last_step)
        )
    END),
    CHECK (blocks_remaining = GREATEST(0, next_epoch_block - current_block)),
    CHECK (snapshot_doc ?& ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ]),
    CHECK (snapshot_doc - ARRAY[
        'schema_version', 'epoch_scheme', 'network_genesis_hash', 'netuid',
        'head_kind', 'block_hash', 'current_block', 'last_epoch_block',
        'pending_epoch_at', 'subnet_epoch_index', 'tempo',
        'blocks_since_last_step', 'observed_at', 'epoch_id', 'epoch_ref',
        'epoch_block', 'next_epoch_block', 'blocks_remaining',
        'settlement_epoch_id', 'cutover_mapping_hash'
    ] = '{}'::JSONB),
    CHECK (snapshot_doc->>'schema_version' = schema_version),
    CHECK (snapshot_doc->>'epoch_scheme' = epoch_scheme),
    CHECK (snapshot_doc->>'network_genesis_hash' = network_genesis_hash),
    CHECK ((snapshot_doc->>'netuid')::INTEGER = netuid),
    CHECK (snapshot_doc->>'head_kind' = head_kind),
    CHECK (snapshot_doc->>'block_hash' = block_hash),
    CHECK ((snapshot_doc->>'current_block')::BIGINT = current_block),
    CHECK ((snapshot_doc->>'last_epoch_block')::BIGINT = last_epoch_block),
    CHECK ((snapshot_doc->>'pending_epoch_at')::BIGINT = pending_epoch_at),
    CHECK ((snapshot_doc->>'subnet_epoch_index')::BIGINT = subnet_epoch_index),
    CHECK ((snapshot_doc->>'tempo')::INTEGER = tempo),
    CHECK ((snapshot_doc->>'blocks_since_last_step')::BIGINT = blocks_since_last_step),
    CHECK ((snapshot_doc->>'observed_at')::TIMESTAMPTZ = observed_at),
    CHECK ((snapshot_doc->>'epoch_id')::BIGINT = subnet_epoch_index),
    CHECK (snapshot_doc->>'epoch_ref' = epoch_ref),
    CHECK ((snapshot_doc->>'epoch_block')::BIGINT = epoch_block),
    CHECK ((snapshot_doc->>'next_epoch_block')::BIGINT = next_epoch_block),
    CHECK ((snapshot_doc->>'blocks_remaining')::BIGINT = blocks_remaining),
    CHECK ((snapshot_doc->>'settlement_epoch_id')::INTEGER = settlement_epoch_id),
    CHECK (snapshot_doc->>'cutover_mapping_hash' = mapping_hash)
);

-- This singleton is the durable operator-controlled write fence.  A cutover
-- is intentionally a multi-runtime deployment, so publishing the authority
-- must not implicitly let legacy writers allocate the new integer namespace.
-- Only the SECURITY DEFINER functions below can mutate this row:
--
--   legacy_open -> cutover_fenced -> stateful_staged -> stateful_active
--
-- ``stateful_staged`` means the cutover authority and the first lifecycle
-- initialization were committed atomically, but all ordinary writes at or
-- above the first settlement ordinal remain blocked.  The operator moves to
-- ``stateful_active`` only after every writer is stopped and the exact
-- stateful release has been prepared offline.  Only that prepared release may
-- then be started; its loaded-code/runtime evidence is verified immediately
-- after activation.
CREATE TABLE IF NOT EXISTS public.research_lab_stateful_subnet_epoch_cutover_state_v1 (
    singleton                                BOOLEAN     PRIMARY KEY DEFAULT TRUE
                                                           CHECK (singleton),
    lifecycle_state                          TEXT        NOT NULL CHECK (
                                                           lifecycle_state IN (
                                                               'legacy_open',
                                                               'cutover_fenced',
                                                               'stateful_staged',
                                                               'stateful_active'
                                                           )
                                                           ),
    mapping_hash                             TEXT        UNIQUE
                                                           CHECK (
                                                               mapping_hash IS NULL
                                                               OR mapping_hash ~ '^sha256:[0-9a-f]{64}$'
                                                           ),
    network_genesis_hash                     TEXT        CHECK (
                                                           network_genesis_hash IS NULL
                                                           OR network_genesis_hash ~ '^0x[0-9a-f]{64}$'
                                                           ),
    netuid                                   INTEGER     CHECK (netuid IS NULL OR netuid > 0),
    last_legacy_epoch_id                     INTEGER     CHECK (
                                                           last_legacy_epoch_id IS NULL
                                                           OR last_legacy_epoch_id >= 0
                                                           ),
    first_settlement_epoch_id                INTEGER     CHECK (
                                                           first_settlement_epoch_id IS NULL
                                                           OR first_settlement_epoch_id >= 0
                                                           ),
    candidate_snapshot_hash                  TEXT        CHECK (
                                                           candidate_snapshot_hash IS NULL
                                                           OR candidate_snapshot_hash ~ '^sha256:[0-9a-f]{64}$'
                                                           ),
    candidate_receipt_hash                   TEXT        CHECK (
                                                           candidate_receipt_hash IS NULL
                                                           OR candidate_receipt_hash ~ '^sha256:[0-9a-f]{64}$'
                                                           ),
    last_legacy_finalization_receipt_hash    TEXT        CHECK (
                                                           last_legacy_finalization_receipt_hash IS NULL
                                                           OR last_legacy_finalization_receipt_hash ~ '^sha256:[0-9a-f]{64}$'
                                                           ),
    cutover_authority_hash                   TEXT        CHECK (
                                                           cutover_authority_hash IS NULL
                                                           OR cutover_authority_hash ~ '^sha256:[0-9a-f]{64}$'
                                                           ),
    cutover_receipt_hash                     TEXT        CHECK (
                                                           cutover_receipt_hash IS NULL
                                                           OR cutover_receipt_hash ~ '^sha256:[0-9a-f]{64}$'
                                                           ),
    initialization_nonce                     UUID,
    initialization_payload_hash              TEXT        CHECK (
                                                           initialization_payload_hash IS NULL
                                                           OR initialization_payload_hash ~ '^[0-9a-f]{64}$'
                                                           ),
    fenced_at                                TIMESTAMPTZ,
    staged_at                                TIMESTAMPTZ,
    activated_at                             TIMESTAMPTZ,
    updated_at                               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (
            lifecycle_state = 'legacy_open'
            AND mapping_hash IS NULL
            AND network_genesis_hash IS NULL
            AND netuid IS NULL
            AND last_legacy_epoch_id IS NULL
            AND first_settlement_epoch_id IS NULL
            AND candidate_snapshot_hash IS NULL
            AND candidate_receipt_hash IS NULL
            AND last_legacy_finalization_receipt_hash IS NULL
            AND cutover_authority_hash IS NULL
            AND cutover_receipt_hash IS NULL
            AND initialization_nonce IS NULL
            AND initialization_payload_hash IS NULL
            AND fenced_at IS NULL
            AND staged_at IS NULL
            AND activated_at IS NULL
        ) OR (
            lifecycle_state IN ('cutover_fenced', 'stateful_staged', 'stateful_active')
            AND network_genesis_hash IS NOT NULL
            AND netuid IS NOT NULL
            AND last_legacy_epoch_id IS NOT NULL
            AND first_settlement_epoch_id = last_legacy_epoch_id + 1
            AND fenced_at IS NOT NULL
            AND (
                (
                    lifecycle_state = 'cutover_fenced'
                    AND mapping_hash IS NULL
                    AND candidate_snapshot_hash IS NULL
                    AND candidate_receipt_hash IS NULL
                    AND last_legacy_finalization_receipt_hash IS NULL
                    AND cutover_authority_hash IS NULL
                ) OR (
                    mapping_hash IS NOT NULL
                    AND candidate_snapshot_hash IS NOT NULL
                    AND candidate_receipt_hash IS NOT NULL
                    AND last_legacy_finalization_receipt_hash IS NOT NULL
                    AND cutover_authority_hash IS NOT NULL
                )
            )
        )
    ),
    CHECK (
        lifecycle_state = 'cutover_fenced'
        OR (
            lifecycle_state IN ('stateful_staged', 'stateful_active')
            AND cutover_receipt_hash IS NOT NULL
            AND initialization_nonce IS NOT NULL
            AND initialization_payload_hash IS NOT NULL
            AND staged_at IS NOT NULL
        )
        OR lifecycle_state = 'legacy_open'
    ),
    CHECK (
        lifecycle_state <> 'stateful_active'
        OR activated_at IS NOT NULL
    )
);

INSERT INTO public.research_lab_stateful_subnet_epoch_cutover_state_v1 (
    singleton, lifecycle_state
) VALUES (TRUE, 'legacy_open')
ON CONFLICT (singleton) DO NOTHING;

-- Validators and auditors do not carry the Supabase service-role credential.
-- Expose only the non-secret singleton fields they need to fail closed on the
-- durable lifecycle state. The protected row remains unreadable directly, and
-- no receipt, candidate, authority document, or mutation surface is exposed.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()
RETURNS TABLE (
    lifecycle_state             TEXT,
    mapping_hash                TEXT,
    network_genesis_hash        TEXT,
    netuid                      INTEGER,
    last_legacy_epoch_id        INTEGER,
    first_settlement_epoch_id   INTEGER,
    fenced_at                   TIMESTAMPTZ,
    staged_at                   TIMESTAMPTZ,
    activated_at                TIMESTAMPTZ,
    updated_at                  TIMESTAMPTZ
)
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT
        state.lifecycle_state,
        state.mapping_hash,
        state.network_genesis_hash,
        state.netuid,
        state.last_legacy_epoch_id,
        state.first_settlement_epoch_id,
        state.fenced_at,
        state.staged_at,
        state.activated_at,
        state.updated_at
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1 state
    WHERE state.singleton = TRUE
$$;

-- Cross-table invariants cannot be represented by foreign keys alone. This
-- trigger verifies that every row is signed by the intended V2 enclave role,
-- that the legacy side really finalized on chain, and that every future
-- mapping is the one affine continuation declared by the cutover manifest.
CREATE OR REPLACE FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    receipt_row RECORD;
    finalization_row RECORD;
    candidate_row RECORD;
    cutover_row RECORD;
    boundary_row RECORD;
    latest_boundary_row RECORD;
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    expected_settlement BIGINT;
BEGIN
    IF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_candidates_v1' THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(
            7099,
            pg_catalog.hashtext(NEW.snapshot_hash)
        );
        SELECT role, purpose, epoch_id, receipt_status, output_root
        INTO receipt_row
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = NEW.chain_state_receipt_hash;
        IF NOT FOUND
           OR receipt_row.role IS DISTINCT FROM 'validator_weights'
           OR receipt_row.purpose IS DISTINCT FROM 'validator.subnet_epoch_snapshot.v2'
           OR receipt_row.epoch_id IS DISTINCT FROM NEW.proposed_settlement_epoch_id
           OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
           OR receipt_row.output_root IS DISTINCT FROM NEW.snapshot_hash THEN
            RAISE EXCEPTION 'stateful epoch candidate snapshot receipt is invalid';
        END IF;

    ELSIF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_cutovers_v1' THEN
        -- The legacy integer namespace is shared by the existing Research Lab
        -- and validator tables.  Serialize all cutovers before taking table
        -- locks and measuring the high-water mark below.
        PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);

        -- Serialize the same epoch_ref across the cutover/boundary tables so
        -- the cross-table reuse check remains race-safe under concurrent inserts.
        PERFORM pg_catalog.pg_advisory_xact_lock(
            7101,
            pg_catalog.hashtext(NEW.first_epoch_ref)
        );

        SELECT role, purpose, epoch_id, receipt_status, output_root
        INTO receipt_row
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = NEW.first_snapshot_receipt_hash;
        IF NOT FOUND
           OR receipt_row.role IS DISTINCT FROM 'validator_weights'
           OR receipt_row.purpose IS DISTINCT FROM 'validator.subnet_epoch_snapshot.v2'
           OR receipt_row.epoch_id IS DISTINCT FROM NEW.first_settlement_epoch_id
           OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
           OR receipt_row.output_root IS DISTINCT FROM NEW.first_snapshot_hash THEN
            RAISE EXCEPTION 'stateful epoch cutover first snapshot receipt is invalid';
        END IF;

        SELECT role, purpose, epoch_id, receipt_status, output_root, receipt_doc
        INTO receipt_row
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = NEW.cutover_receipt_hash;
        IF NOT FOUND
           OR receipt_row.role IS DISTINCT FROM 'gateway_coordinator'
           OR receipt_row.purpose IS DISTINCT FROM 'research_lab.subnet_epoch_cutover.v2'
           OR receipt_row.epoch_id IS DISTINCT FROM NEW.first_settlement_epoch_id
           OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
           OR receipt_row.output_root IS DISTINCT FROM NEW.cutover_authority_hash
           OR receipt_row.receipt_doc->>'receipt_hash' IS DISTINCT FROM NEW.cutover_receipt_hash
           OR receipt_row.receipt_doc->>'role' IS DISTINCT FROM 'gateway_coordinator'
           OR receipt_row.receipt_doc->>'purpose' IS DISTINCT FROM 'research_lab.subnet_epoch_cutover.v2'
           OR (receipt_row.receipt_doc->>'epoch_id')::INTEGER IS DISTINCT FROM NEW.first_settlement_epoch_id
           OR receipt_row.receipt_doc->>'output_root' IS DISTINCT FROM NEW.cutover_authority_hash
           OR receipt_row.receipt_doc->>'status' IS DISTINCT FROM 'succeeded'
           OR pg_catalog.jsonb_typeof(
                  receipt_row.receipt_doc->'parent_receipt_hashes'
              ) IS DISTINCT FROM 'array'
           OR NOT (
               receipt_row.receipt_doc->'parent_receipt_hashes'
               @> pg_catalog.jsonb_build_array(
                   NEW.first_snapshot_receipt_hash,
                   NEW.last_legacy_finalization_receipt_hash
               )
           ) THEN
            RAISE EXCEPTION 'stateful epoch cutover coordinator receipt is invalid';
        END IF;
        IF pg_catalog.jsonb_array_length(
               receipt_row.receipt_doc->'parent_receipt_hashes'
           ) IS DISTINCT FROM 2 THEN
            RAISE EXCEPTION 'stateful epoch cutover coordinator receipt is invalid';
        END IF;

        SELECT
            bundle.netuid,
            bundle.epoch_id,
            finalization.bundle_hash,
            finalization.finalization_receipt_hash,
            finalization.finalized_block
        INTO finalization_row
        FROM public.research_lab_attested_weight_finalizations_v2 finalization
        JOIN public.research_lab_attested_weight_bundles_v2 bundle
          ON bundle.bundle_hash = finalization.bundle_hash
        JOIN public.research_lab_attested_publication_events_v2 publication
          ON publication.weight_submission_event_hash =
             finalization.weight_submission_event_hash
         AND publication.bundle_hash = finalization.bundle_hash
        WHERE finalization.weight_finalization_event_hash =
              NEW.last_legacy_weight_finalization_event_hash;
        IF NOT FOUND
           OR finalization_row.netuid IS DISTINCT FROM NEW.netuid
           OR finalization_row.epoch_id IS DISTINCT FROM NEW.last_legacy_epoch_id
           OR finalization_row.bundle_hash IS DISTINCT FROM NEW.last_legacy_bundle_hash
           OR finalization_row.finalization_receipt_hash IS DISTINCT FROM
              NEW.last_legacy_finalization_receipt_hash
           -- The legacy-global scheduler leads the official SN71 boundary by
           -- 36 blocks in the observed cutover schedule.  Its exact finalization
           -- therefore must not be after the official boundary; equality is
           -- valid if finalization lands in the boundary block itself.
           OR finalization_row.finalized_block > NEW.cutover_block THEN
            RAISE EXCEPTION 'stateful epoch cutover legacy finalization proof is invalid';
        END IF;

        SELECT role, purpose, epoch_id, receipt_status
        INTO receipt_row
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = NEW.last_legacy_finalization_receipt_hash;
        IF NOT FOUND
           OR receipt_row.role IS DISTINCT FROM 'validator_weights'
           OR receipt_row.purpose IS DISTINCT FROM 'validator.weights.finalized.v2'
           OR receipt_row.epoch_id IS DISTINCT FROM NEW.last_legacy_epoch_id
           OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded' THEN
            RAISE EXCEPTION 'stateful epoch cutover legacy finalization receipt is invalid';
        END IF;

        SELECT *
        INTO candidate_row
        FROM public.research_lab_stateful_subnet_epoch_candidates_v1
        WHERE snapshot_hash = NEW.first_snapshot_hash;
        IF NOT FOUND
           OR candidate_row.mapping_hash IS DISTINCT FROM NEW.mapping_hash
           OR candidate_row.epoch_scheme IS DISTINCT FROM NEW.epoch_scheme
           OR candidate_row.network_genesis_hash IS DISTINCT FROM NEW.network_genesis_hash
           OR candidate_row.netuid IS DISTINCT FROM NEW.netuid
           OR candidate_row.current_block IS DISTINCT FROM NEW.cutover_block
           OR candidate_row.last_epoch_block IS DISTINCT FROM NEW.cutover_block
           OR candidate_row.block_hash IS DISTINCT FROM NEW.cutover_block_hash
           OR candidate_row.subnet_epoch_index IS DISTINCT FROM NEW.first_subnet_epoch_index
           OR candidate_row.epoch_ref IS DISTINCT FROM NEW.first_epoch_ref
           OR candidate_row.proposed_settlement_epoch_id IS DISTINCT FROM NEW.first_settlement_epoch_id
           OR candidate_row.tempo IS DISTINCT FROM NEW.first_tempo
           OR candidate_row.pending_epoch_at IS DISTINCT FROM NEW.first_pending_epoch_at
           OR candidate_row.blocks_since_last_step IS DISTINCT FROM NEW.first_blocks_since_last_step
           OR candidate_row.next_epoch_block IS DISTINCT FROM NEW.first_next_epoch_block
           OR candidate_row.observed_at IS DISTINCT FROM NEW.first_observed_at
           OR candidate_row.chain_state_receipt_hash IS DISTINCT FROM NEW.first_snapshot_receipt_hash
           OR candidate_row.snapshot_doc IS DISTINCT FROM NEW.first_snapshot_doc THEN
            RAISE EXCEPTION 'stateful epoch cutover first candidate snapshot differs';
        END IF;

        -- The namespace was measured once, under SHARE locks, by the early
        -- fence RPC.  Its generic triggers then reject every ordinary identity
        -- at or above the first settlement ordinal.  Re-scanning every legacy
        -- table here is both redundant and unsafe for large production tables;
        -- bind this insert to the durable fenced plan instead.
        SELECT * INTO state_row
        FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
        WHERE singleton
        FOR SHARE;
        IF NOT FOUND
           OR state_row.lifecycle_state NOT IN (
               'cutover_fenced', 'stateful_staged', 'stateful_active'
           )
           OR state_row.mapping_hash IS DISTINCT FROM NEW.mapping_hash
           OR state_row.network_genesis_hash IS DISTINCT FROM
              NEW.network_genesis_hash
           OR state_row.netuid IS DISTINCT FROM NEW.netuid
           OR state_row.last_legacy_epoch_id IS DISTINCT FROM
              NEW.last_legacy_epoch_id
           OR state_row.first_settlement_epoch_id IS DISTINCT FROM
              NEW.first_settlement_epoch_id
           OR state_row.candidate_snapshot_hash IS DISTINCT FROM
              NEW.first_snapshot_hash
           OR state_row.candidate_receipt_hash IS DISTINCT FROM
              NEW.first_snapshot_receipt_hash
           OR state_row.last_legacy_finalization_receipt_hash IS DISTINCT FROM
              NEW.last_legacy_finalization_receipt_hash
           OR state_row.cutover_authority_hash IS DISTINCT FROM
              NEW.cutover_authority_hash
           OR state_row.cutover_receipt_hash IS DISTINCT FROM
              NEW.cutover_receipt_hash THEN
            RAISE EXCEPTION 'stateful epoch cutover differs from durable fence';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM public.research_lab_stateful_subnet_epoch_boundaries_v1 boundary
            WHERE boundary.epoch_ref = NEW.first_epoch_ref
        ) THEN
            RAISE EXCEPTION 'stateful epoch cutover epoch_ref is already mapped';
        END IF;

    ELSIF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_boundaries_v1' THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(
            7101,
            pg_catalog.hashtext(NEW.epoch_ref)
        );

        -- Lock the cutover row to serialize every boundary in this lineage.
        -- Without it, two otherwise-valid concurrent inserts could both see
        -- the same latest accepted boundary.
        SELECT *
        INTO cutover_row
        FROM public.research_lab_stateful_subnet_epoch_cutovers_v1
        WHERE mapping_hash = NEW.mapping_hash
        FOR UPDATE;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'stateful epoch boundary has no cutover mapping';
        END IF;
        expected_settlement := cutover_row.first_settlement_epoch_id::BIGINT
            + (NEW.subnet_epoch_index - cutover_row.first_subnet_epoch_index);
        IF NEW.epoch_scheme IS DISTINCT FROM cutover_row.epoch_scheme
           OR NEW.network_genesis_hash IS DISTINCT FROM cutover_row.network_genesis_hash
           OR NEW.netuid IS DISTINCT FROM cutover_row.netuid
           OR NEW.subnet_epoch_index <= cutover_row.first_subnet_epoch_index
           OR NEW.boundary_block <= cutover_row.cutover_block
           OR expected_settlement < 0
           OR expected_settlement > 2147483647
           OR NEW.settlement_epoch_id::BIGINT IS DISTINCT FROM expected_settlement THEN
            RAISE EXCEPTION 'stateful epoch boundary mapping differs from cutover';
        END IF;

        IF EXISTS (
            SELECT 1
            FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 other_cutover
            WHERE other_cutover.first_epoch_ref = NEW.epoch_ref
        ) THEN
            RAISE EXCEPTION 'stateful epoch boundary epoch_ref is already mapped';
        END IF;

        -- Boundary rows are an append-only observed sequence.  A validator may
        -- miss an entire weight epoch, so the next authenticated boundary is
        -- allowed to jump forward.  It must still advance both the official
        -- index and finalized boundary block from the latest durable boundary.
        -- Once a jump is accepted, a historical gap can never be filled.
        SELECT subnet_epoch_index, settlement_epoch_id, boundary_block
        INTO latest_boundary_row
        FROM public.research_lab_stateful_subnet_epoch_boundaries_v1
        WHERE mapping_hash = NEW.mapping_hash
        ORDER BY subnet_epoch_index DESC
        LIMIT 1;
        IF FOUND THEN
            IF NEW.subnet_epoch_index <= latest_boundary_row.subnet_epoch_index
               OR NEW.settlement_epoch_id::BIGINT <=
                  latest_boundary_row.settlement_epoch_id::BIGINT
               OR NEW.boundary_block <= latest_boundary_row.boundary_block THEN
                RAISE EXCEPTION 'stateful epoch boundary does not advance the latest accepted boundary';
            END IF;
        ELSIF NEW.boundary_block <= cutover_row.cutover_block THEN
            RAISE EXCEPTION 'stateful epoch boundary block does not follow the cutover boundary';
        END IF;

        SELECT role, purpose, epoch_id, receipt_status, output_root
        INTO receipt_row
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = NEW.chain_state_receipt_hash;
        IF NOT FOUND
           OR receipt_row.role IS DISTINCT FROM 'validator_weights'
           OR receipt_row.purpose IS DISTINCT FROM 'validator.subnet_epoch_snapshot.v2'
           OR receipt_row.epoch_id IS DISTINCT FROM NEW.settlement_epoch_id
           OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
           OR receipt_row.output_root IS DISTINCT FROM NEW.boundary_hash THEN
            RAISE EXCEPTION 'stateful epoch boundary chain-state receipt is invalid';
        END IF;

    ELSIF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_snapshots_v1' THEN
        SELECT *
        INTO cutover_row
        FROM public.research_lab_stateful_subnet_epoch_cutovers_v1
        WHERE mapping_hash = NEW.mapping_hash;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'stateful epoch snapshot has no cutover mapping';
        END IF;
        expected_settlement := cutover_row.first_settlement_epoch_id::BIGINT
            + (NEW.subnet_epoch_index - cutover_row.first_subnet_epoch_index);
        IF NEW.epoch_scheme IS DISTINCT FROM cutover_row.epoch_scheme
           OR NEW.network_genesis_hash IS DISTINCT FROM cutover_row.network_genesis_hash
           OR NEW.netuid IS DISTINCT FROM cutover_row.netuid
           OR NEW.subnet_epoch_index < cutover_row.first_subnet_epoch_index
           OR expected_settlement < 0
           OR expected_settlement > 2147483647
           OR NEW.settlement_epoch_id::BIGINT IS DISTINCT FROM expected_settlement THEN
            RAISE EXCEPTION 'stateful epoch snapshot mapping differs from cutover';
        END IF;

        IF NEW.subnet_epoch_index = cutover_row.first_subnet_epoch_index THEN
            IF NEW.epoch_ref IS DISTINCT FROM cutover_row.first_epoch_ref
               OR NEW.current_block < cutover_row.cutover_block THEN
                RAISE EXCEPTION 'stateful epoch snapshot differs from first cutover epoch';
            END IF;
        ELSE
            SELECT epoch_ref, settlement_epoch_id, boundary_block
            INTO boundary_row
            FROM public.research_lab_stateful_subnet_epoch_boundaries_v1
            WHERE mapping_hash = NEW.mapping_hash
              AND subnet_epoch_index = NEW.subnet_epoch_index;
            IF NOT FOUND
               OR boundary_row.epoch_ref IS DISTINCT FROM NEW.epoch_ref
               OR boundary_row.settlement_epoch_id IS DISTINCT FROM NEW.settlement_epoch_id
               OR NEW.current_block < boundary_row.boundary_block THEN
                RAISE EXCEPTION 'stateful epoch snapshot has no matching boundary mapping';
            END IF;
        END IF;

        SELECT role, purpose, epoch_id, receipt_status, output_root
        INTO receipt_row
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = NEW.chain_state_receipt_hash;
        IF NOT FOUND
           OR receipt_row.role IS DISTINCT FROM 'validator_weights'
           OR receipt_row.purpose IS DISTINCT FROM 'validator.subnet_epoch_snapshot.v2'
           OR receipt_row.epoch_id IS DISTINCT FROM NEW.settlement_epoch_id
           OR receipt_row.receipt_status IS DISTINCT FROM 'succeeded'
           OR receipt_row.output_root IS DISTINCT FROM NEW.snapshot_hash THEN
            RAISE EXCEPTION 'stateful epoch snapshot chain-state receipt is invalid';
        END IF;
    ELSE
        RAISE EXCEPTION 'stateful subnet epoch validator attached to unexpected table %', TG_TABLE_NAME;
    END IF;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1()
    TO service_role;

-- Read-only operator preflight for the explicit cutover CLI. The early fence
-- already measured and locked the namespace once, and its triggers make that
-- result durable. This RPC therefore validates the candidate/optional resumed
-- coordinator receipt against the fenced plan without rescanning production.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_preflight_v1(
    p_mapping_hash TEXT,
    p_cutover_receipt_hash TEXT DEFAULT NULL
)
RETURNS TABLE (
    eligible                         BOOLEAN,
    legacy_high_water                BIGINT,
    expected_last_legacy_epoch_id    BIGINT,
    first_settlement_epoch_id        BIGINT,
    first_settlement_occupied        BOOLEAN,
    candidate_snapshot_hash          TEXT,
    candidate_receipt_hash           TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    candidate_row RECORD;
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    coordinator_receipt RECORD;
BEGIN
    IF p_mapping_hash !~ '^sha256:[0-9a-f]{64}$'
       OR (
           p_cutover_receipt_hash IS NOT NULL
           AND p_cutover_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       ) THEN
        RAISE EXCEPTION 'stateful epoch cutover preflight hash is invalid';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR SHARE;
    IF NOT FOUND
       OR state_row.lifecycle_state NOT IN (
           'cutover_fenced', 'stateful_staged', 'stateful_active'
       ) THEN
        RAISE EXCEPTION 'stateful epoch cutover preflight requires the durable fence';
    END IF;

    SELECT
        candidate.mapping_hash,
        candidate.network_genesis_hash,
        candidate.netuid,
        candidate.proposed_settlement_epoch_id,
        candidate.snapshot_hash,
        candidate.chain_state_receipt_hash
    INTO candidate_row
    FROM public.research_lab_stateful_subnet_epoch_candidates_v1 AS candidate
    WHERE candidate.mapping_hash = p_mapping_hash;
    IF NOT FOUND
       OR candidate_row.network_genesis_hash IS DISTINCT FROM
          state_row.network_genesis_hash
       OR candidate_row.netuid IS DISTINCT FROM state_row.netuid
       OR candidate_row.proposed_settlement_epoch_id IS DISTINCT FROM
          state_row.first_settlement_epoch_id
       OR (
           state_row.mapping_hash IS NOT NULL
           AND (
               state_row.mapping_hash IS DISTINCT FROM candidate_row.mapping_hash
               OR state_row.candidate_snapshot_hash IS DISTINCT FROM
                  candidate_row.snapshot_hash
               OR state_row.candidate_receipt_hash IS DISTINCT FROM
                  candidate_row.chain_state_receipt_hash
           )
       ) THEN
        RAISE EXCEPTION 'stateful epoch cutover preflight candidate is missing';
    END IF;

    IF p_cutover_receipt_hash IS NOT NULL THEN
        SELECT receipt.role, receipt.purpose, receipt.epoch_id,
               receipt.receipt_status, receipt.output_root
        INTO coordinator_receipt
        FROM public.research_lab_attested_execution_receipts_v2 AS receipt
        WHERE receipt.receipt_hash = p_cutover_receipt_hash;
        IF NOT FOUND
           OR state_row.mapping_hash IS DISTINCT FROM p_mapping_hash
           OR coordinator_receipt.role IS DISTINCT FROM 'gateway_coordinator'
           OR coordinator_receipt.purpose IS DISTINCT FROM
              'research_lab.subnet_epoch_cutover.v2'
           OR coordinator_receipt.epoch_id IS DISTINCT FROM
              state_row.first_settlement_epoch_id
           OR coordinator_receipt.receipt_status IS DISTINCT FROM 'succeeded'
           OR coordinator_receipt.output_root IS DISTINCT FROM
              state_row.cutover_authority_hash THEN
            RAISE EXCEPTION 'stateful epoch cutover preflight receipt is invalid';
        END IF;
    END IF;

    RETURN QUERY SELECT
        TRUE,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.first_settlement_epoch_id::BIGINT,
        FALSE,
        candidate_row.snapshot_hash,
        candidate_row.chain_state_receipt_hash;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_preflight_v1(TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_preflight_v1(TEXT, TEXT)
    TO service_role;
COMMENT ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_preflight_v1(TEXT, TEXT) IS
    'Read-only durable-fence preflight for the explicit receipt-backed stateful subnet epoch cutover. Insert-time triggers remain authoritative.';

-- Measure the exact pre-cutover namespace with the same identity scope used by
-- the durable fence. This is service-role-only and read-only; the fence still
-- repeats the measurement while holding relation locks before it writes state.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_legacy_high_water_v1()
RETURNS BIGINT
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
SET statement_timeout = '120s'
AS $$
DECLARE
    epoch_column RECORD;
    column_max BIGINT;
    measured_high_water BIGINT;
BEGIN
    FOR epoch_column IN
        SELECT c.oid AS relation_oid, n.nspname, c.relname,
               a.attname, a.attnum, a.atttypid
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND a.atttypid IN (20, 21, 23)
          AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
          AND c.relname NOT IN (
              'research_lab_stateful_subnet_epoch_cutover_state_v1',
              'research_lab_stateful_subnet_epoch_cutovers_v1',
              'research_lab_stateful_subnet_epoch_boundaries_v1',
              'research_lab_stateful_subnet_epoch_snapshots_v1'
          )
        ORDER BY c.oid, a.attnum
    LOOP
        EXECUTE pg_catalog.format(
            'SELECT %1$I::BIGINT FROM %2$I.%3$I '
            'WHERE %1$I IS NOT NULL ORDER BY %1$I DESC LIMIT 1',
            epoch_column.attname,
            epoch_column.nspname,
            epoch_column.relname
        )
        INTO column_max;
        IF column_max IS NOT NULL
           AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
            measured_high_water := column_max;
        END IF;
    END LOOP;

    SELECT (payload->>'epoch_id')::BIGINT
    INTO column_max
    FROM public.transparency_log
    WHERE pg_catalog.jsonb_typeof(payload) = 'object'
      AND payload ? 'epoch_id'
      AND payload->>'epoch_id' ~ '^[0-9]+$'
      AND payload->>'epoch_key_semantics' = 'legacy_global_360'
    ORDER BY (payload->>'epoch_id')::BIGINT DESC
    LIMIT 1;
    IF column_max IS NOT NULL
       AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
        measured_high_water := column_max;
    END IF;

    IF measured_high_water IS NULL THEN
        RAISE EXCEPTION 'stateful epoch legacy high-water is unavailable';
    END IF;
    RETURN measured_high_water;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_legacy_high_water_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_legacy_high_water_v1()
    TO service_role;


-- Reserve the next physical integer ordinal well before the legacy-global
-- boundary and before the official subnet boundary/candidate exist. Legacy
-- writers may continue writing the measured last legacy ordinal; the installed
-- generic triggers reject the reserved first ordinal. SHARE locks make the
-- high-water/vacancy proof remain true until the durable fence row commits,
-- after which those triggers protect the offset gap between schedulers.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(
    p_network_genesis_hash TEXT,
    p_netuid INTEGER,
    p_last_legacy_epoch_id INTEGER,
    p_first_settlement_epoch_id INTEGER
)
RETURNS TABLE (
    lifecycle_state                     TEXT,
    network_genesis_hash                TEXT,
    netuid                              INTEGER,
    legacy_high_water                   BIGINT,
    last_legacy_epoch_id                BIGINT,
    first_settlement_epoch_id           BIGINT,
    first_settlement_occupied           BOOLEAN,
    fenced_at                           TIMESTAMPTZ
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
SET lock_timeout = '5s'
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    epoch_relation RECORD;
    epoch_column RECORD;
    column_max BIGINT;
    column_occupied BOOLEAN;
    measured_high_water BIGINT := NULL;
    measured_occupied BOOLEAN := FALSE;
BEGIN
    IF p_network_genesis_hash !~ '^0x[0-9a-f]{64}$'
       OR p_netuid <= 0
       OR p_last_legacy_epoch_id < 0
       OR p_first_settlement_epoch_id::BIGINT IS DISTINCT FROM
          p_last_legacy_epoch_id::BIGINT + 1 THEN
        RAISE EXCEPTION 'stateful epoch pre-boundary fence input is invalid';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'stateful epoch cutover singleton state is missing';
    END IF;

    IF state_row.lifecycle_state <> 'legacy_open' THEN
        IF state_row.network_genesis_hash IS DISTINCT FROM p_network_genesis_hash
           OR state_row.netuid IS DISTINCT FROM p_netuid
           OR state_row.last_legacy_epoch_id IS DISTINCT FROM
              p_last_legacy_epoch_id
           OR state_row.first_settlement_epoch_id IS DISTINCT FROM
              p_first_settlement_epoch_id THEN
            RAISE EXCEPTION 'stateful epoch pre-boundary fence conflicts with durable plan';
        END IF;
        RETURN QUERY SELECT
            state_row.lifecycle_state,
            state_row.network_genesis_hash,
            state_row.netuid,
            state_row.last_legacy_epoch_id::BIGINT,
            state_row.last_legacy_epoch_id::BIGINT,
            state_row.first_settlement_epoch_id::BIGINT,
            FALSE,
            state_row.fenced_at;
        RETURN;
    END IF;

    -- The payload epoch identity lives inside JSONB and cannot use a normal
    -- column index.  Require the separately installed concurrent expression
    -- index before taking any production locks or attempting the measurement.
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_index index_meta
        JOIN pg_catalog.pg_class index_relation
          ON index_relation.oid = index_meta.indexrelid
        JOIN pg_catalog.pg_namespace index_namespace
          ON index_namespace.oid = index_relation.relnamespace
        JOIN pg_catalog.pg_class table_relation
          ON table_relation.oid = index_meta.indrelid
        JOIN pg_catalog.pg_namespace table_namespace
          ON table_namespace.oid = table_relation.relnamespace
        JOIN pg_catalog.pg_attribute payload_column
          ON payload_column.attrelid = table_relation.oid
         AND payload_column.attname = 'payload'
        JOIN pg_catalog.pg_am access_method
          ON access_method.oid = index_relation.relam
        JOIN pg_catalog.pg_opclass operator_class
          ON operator_class.oid = index_meta.indclass[0]
        JOIN pg_catalog.pg_namespace operator_class_namespace
          ON operator_class_namespace.oid = operator_class.opcnamespace
        WHERE index_meta.indrelid =
              'public.transparency_log'::pg_catalog.regclass
          AND table_namespace.nspname = 'public'
          AND table_relation.relname = 'transparency_log'
          AND table_relation.relkind = 'r'
          AND payload_column.attnum > 0
          AND NOT payload_column.attisdropped
          AND payload_column.atttypid =
              'pg_catalog.jsonb'::pg_catalog.regtype
          AND index_namespace.nspname = 'public'
          AND index_relation.relname =
              'idx_transparency_log_payload_epoch_identity_v1'
          AND index_relation.relkind = 'i'
          AND index_relation.relpersistence = table_relation.relpersistence
          AND access_method.amname = 'btree'
          AND index_meta.indisvalid
          AND index_meta.indisready
          AND index_meta.indislive
          AND NOT index_meta.indisunique
          AND NOT index_meta.indisprimary
          AND NOT index_meta.indisexclusion
          AND index_meta.indnatts = 1
          AND index_meta.indnkeyatts = 1
          AND index_meta.indkey[0] = 0
          AND index_meta.indoption[0] = 3
          AND index_meta.indcollation[0] = 0
          AND operator_class.opcdefault
          AND operator_class.opcmethod = index_relation.relam
          AND operator_class.opcintype =
              'pg_catalog.int8'::pg_catalog.regtype
          AND operator_class.opcname = 'int8_ops'
          AND operator_class_namespace.nspname = 'pg_catalog'
          AND index_meta.indexprs IS NOT NULL
          AND index_meta.indpred IS NOT NULL
          AND pg_catalog.pg_get_expr(
                  index_meta.indexprs,
                  index_meta.indrelid,
                  FALSE
              ) = '((payload ->> ''epoch_id''::text))::bigint'
          AND pg_catalog.pg_get_expr(
                  index_meta.indpred,
                  index_meta.indrelid,
                  FALSE
              ) = '((jsonb_typeof(payload) = ''object''::text) AND (payload ? ''epoch_id''::text) AND ((payload ->> ''epoch_id''::text) ~ ''^[0-9]+$''::text))'
    ) THEN
        RAISE EXCEPTION
            'stateful epoch fence prerequisite payload identity index is missing or invalid';
    END IF;

    FOR epoch_relation IN
        SELECT DISTINCT c.oid, n.nspname, c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND (
              (
                  a.atttypid IN (20, 21, 23)
                  AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
              )
              OR c.relname = 'transparency_log'
          )
          AND c.relname NOT IN (
              'research_lab_stateful_subnet_epoch_cutover_state_v1',
              'research_lab_stateful_subnet_epoch_cutovers_v1',
              'research_lab_stateful_subnet_epoch_boundaries_v1',
              'research_lab_stateful_subnet_epoch_snapshots_v1'
          )
        ORDER BY c.oid
    LOOP
        -- Migration-time trigger installation avoids ACCESS EXCLUSIVE DDL in
        -- the live reservation RPC. A table added after the migration is an
        -- explicit schema-drift failure, never an excuse to reserve an
        -- incompletely guarded namespace.
        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_trigger trigger_meta
            JOIN pg_catalog.pg_proc trigger_function
              ON trigger_function.oid = trigger_meta.tgfoid
            JOIN pg_catalog.pg_namespace function_namespace
              ON function_namespace.oid = trigger_function.pronamespace
            WHERE trigger_meta.tgrelid = epoch_relation.oid
              AND NOT trigger_meta.tgisinternal
              AND trigger_meta.tgenabled <> 'D'
              AND function_namespace.nspname = 'public'
              AND trigger_function.proname =
                  'enforce_research_lab_stateful_epoch_fence_v1'
              AND (trigger_meta.tgtype & 1) = 1
              AND (trigger_meta.tgtype & 2) = 2
              AND (trigger_meta.tgtype & 4) = 4
              AND (trigger_meta.tgtype & 16) = 16
        ) THEN
            RAISE EXCEPTION
                'stateful epoch fence trigger is missing for %',
                epoch_relation.relname;
        END IF;
        EXECUTE pg_catalog.format(
            'LOCK TABLE %I.%I IN SHARE MODE',
            epoch_relation.nspname,
            epoch_relation.relname
        );
    END LOOP;

    FOR epoch_column IN
        -- Only physical identity columns participate here; the separate
        -- Explicitly typed legacy-global transparency identities are included
        -- below. Unscoped historical payloads can belong to another network
        -- and must not define this Finney SN71 namespace. In particular,
        -- reward_expires_epoch must not raise the high-water mark because it
        -- is a future schedule/reference, not an allocated epoch identity.
        SELECT c.oid AS relation_oid, n.nspname, c.relname,
               a.attname, a.attnum, a.atttypid
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p')
          AND a.attnum > 0
          AND NOT a.attisdropped
          AND a.atttypid IN (20, 21, 23)
          AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
          AND c.relname NOT IN (
              'research_lab_stateful_subnet_epoch_cutover_state_v1',
              'research_lab_stateful_subnet_epoch_cutovers_v1',
              'research_lab_stateful_subnet_epoch_boundaries_v1',
              'research_lab_stateful_subnet_epoch_snapshots_v1'
          )
        ORDER BY c.oid, a.attnum
    LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_index index_meta
            JOIN pg_catalog.pg_class index_relation
              ON index_relation.oid = index_meta.indexrelid
            JOIN pg_catalog.pg_am access_method
              ON access_method.oid = index_relation.relam
            JOIN pg_catalog.pg_opclass operator_class
              ON operator_class.oid = index_meta.indclass[0]
            WHERE index_meta.indrelid = epoch_column.relation_oid
              AND index_relation.relkind IN ('i', 'I')
              AND access_method.amname = 'btree'
              AND index_meta.indisvalid
              AND index_meta.indisready
              AND index_meta.indislive
              AND index_meta.indpred IS NULL
              AND index_meta.indexprs IS NULL
              AND index_meta.indnkeyatts >= 1
              AND index_meta.indkey[0] = epoch_column.attnum
              AND operator_class.opcdefault
              AND operator_class.opcmethod = index_relation.relam
              AND operator_class.opcintype = epoch_column.atttypid
        ) THEN
            RAISE EXCEPTION
                'stateful epoch fence prerequisite btree index is missing for %.%',
                epoch_column.relname,
                epoch_column.attname;
        END IF;

        EXECUTE pg_catalog.format(
            'SELECT %1$I::BIGINT FROM %2$I.%3$I '
            'WHERE %1$I IS NOT NULL ORDER BY %1$I DESC LIMIT 1',
            epoch_column.attname,
            epoch_column.nspname,
            epoch_column.relname
        )
        INTO column_max;
        EXECUTE pg_catalog.format(
            'SELECT EXISTS (SELECT 1 FROM %2$I.%3$I WHERE %1$I = $1)',
            epoch_column.attname,
            epoch_column.nspname,
            epoch_column.relname
        )
        INTO column_occupied
        USING p_first_settlement_epoch_id;
        IF column_max IS NOT NULL
           AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
            measured_high_water := column_max;
        END IF;
        measured_occupied := measured_occupied OR COALESCE(column_occupied, FALSE);
    END LOOP;

    SELECT (payload->>'epoch_id')::BIGINT
    INTO column_max
    FROM public.transparency_log
    WHERE pg_catalog.jsonb_typeof(payload) = 'object'
      AND payload ? 'epoch_id'
      AND payload->>'epoch_id' ~ '^[0-9]+$'
      AND payload->>'epoch_key_semantics' = 'legacy_global_360'
    ORDER BY (payload->>'epoch_id')::BIGINT DESC
    LIMIT 1;
    SELECT EXISTS (
        SELECT 1
        FROM public.transparency_log
        WHERE pg_catalog.jsonb_typeof(payload) = 'object'
          AND payload ? 'epoch_id'
          AND payload->>'epoch_id' ~ '^[0-9]+$'
          AND payload->>'epoch_key_semantics' = 'legacy_global_360'
          AND (payload->>'epoch_id')::BIGINT = p_first_settlement_epoch_id
    )
    INTO column_occupied;
    IF column_max IS NOT NULL
       AND (measured_high_water IS NULL OR column_max > measured_high_water) THEN
        measured_high_water := column_max;
    END IF;
    measured_occupied := measured_occupied OR COALESCE(column_occupied, FALSE);

    IF measured_occupied
       OR measured_high_water IS DISTINCT FROM p_last_legacy_epoch_id::BIGINT THEN
        RAISE EXCEPTION
            'stateful epoch pre-boundary fence expected high-water %, observed %, occupied %',
            p_last_legacy_epoch_id, measured_high_water, measured_occupied;
    END IF;

    UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    SET lifecycle_state = 'cutover_fenced',
        network_genesis_hash = p_network_genesis_hash,
        netuid = p_netuid,
        last_legacy_epoch_id = p_last_legacy_epoch_id,
        first_settlement_epoch_id = p_first_settlement_epoch_id,
        fenced_at = pg_catalog.clock_timestamp(),
        updated_at = pg_catalog.clock_timestamp()
    WHERE singleton
    RETURNING * INTO state_row;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.network_genesis_hash,
        state_row.netuid,
        measured_high_water,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.first_settlement_epoch_id::BIGINT,
        measured_occupied,
        state_row.fenced_at;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(TEXT, INTEGER, INTEGER, INTEGER)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_fence_v1(TEXT, INTEGER, INTEGER, INTEGER)
    TO service_role;

-- After the official boundary candidate and the last legacy finalization are
-- durable, bind them to the already-closed namespace.  This never opens the
-- fence and cannot change the pre-boundary integer plan.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_cutover_bind_v1(
    p_mapping_hash TEXT,
    p_cutover_authority_hash TEXT,
    p_last_legacy_finalization_receipt_hash TEXT,
    p_cutover_receipt_hash TEXT DEFAULT NULL
)
RETURNS TABLE (
    lifecycle_state                     TEXT,
    mapping_hash                        TEXT,
    legacy_high_water                   BIGINT,
    last_legacy_epoch_id                BIGINT,
    first_settlement_epoch_id           BIGINT,
    candidate_snapshot_hash             TEXT,
    candidate_receipt_hash              TEXT,
    cutover_authority_hash              TEXT,
    cutover_receipt_hash                TEXT,
    initialization_nonce                UUID,
    initialization_payload_hash         TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    candidate_row RECORD;
    finalization_receipt RECORD;
    coordinator_receipt RECORD;
BEGIN
    IF p_mapping_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_cutover_authority_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_last_legacy_finalization_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR (
           p_cutover_receipt_hash IS NOT NULL
           AND p_cutover_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       ) THEN
        RAISE EXCEPTION 'stateful epoch cutover binding hash is invalid';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND OR state_row.lifecycle_state <> 'cutover_fenced' THEN
        RAISE EXCEPTION 'stateful epoch cutover binding requires the pre-boundary fence';
    END IF;

    SELECT candidate.* INTO candidate_row
    FROM public.research_lab_stateful_subnet_epoch_candidates_v1 AS candidate
    WHERE candidate.mapping_hash = p_mapping_hash;
    IF NOT FOUND
       OR candidate_row.network_genesis_hash IS DISTINCT FROM
          state_row.network_genesis_hash
       OR candidate_row.netuid IS DISTINCT FROM state_row.netuid
       OR candidate_row.proposed_settlement_epoch_id IS DISTINCT FROM
          state_row.first_settlement_epoch_id THEN
        RAISE EXCEPTION 'stateful epoch cutover binding candidate differs from fence';
    END IF;

    SELECT role, purpose, epoch_id, receipt_status
    INTO finalization_receipt
    FROM public.research_lab_attested_execution_receipts_v2
    WHERE receipt_hash = p_last_legacy_finalization_receipt_hash;
    IF NOT FOUND
       OR finalization_receipt.role IS DISTINCT FROM 'validator_weights'
       OR finalization_receipt.purpose IS DISTINCT FROM 'validator.weights.finalized.v2'
       OR finalization_receipt.epoch_id IS DISTINCT FROM state_row.last_legacy_epoch_id
       OR finalization_receipt.receipt_status IS DISTINCT FROM 'succeeded' THEN
        RAISE EXCEPTION 'stateful epoch cutover binding finalization receipt is invalid';
    END IF;

    IF p_cutover_receipt_hash IS NOT NULL THEN
        SELECT role, purpose, epoch_id, receipt_status, output_root, receipt_doc
        INTO coordinator_receipt
        FROM public.research_lab_attested_execution_receipts_v2
        WHERE receipt_hash = p_cutover_receipt_hash;
        IF NOT FOUND
           OR coordinator_receipt.role IS DISTINCT FROM 'gateway_coordinator'
           OR coordinator_receipt.purpose IS DISTINCT FROM
              'research_lab.subnet_epoch_cutover.v2'
           OR coordinator_receipt.epoch_id IS DISTINCT FROM
              state_row.first_settlement_epoch_id
           OR coordinator_receipt.receipt_status IS DISTINCT FROM 'succeeded'
           OR coordinator_receipt.output_root IS DISTINCT FROM
              p_cutover_authority_hash
           OR pg_catalog.jsonb_typeof(
                  coordinator_receipt.receipt_doc->'parent_receipt_hashes'
              ) IS DISTINCT FROM 'array'
           OR pg_catalog.jsonb_array_length(
                  coordinator_receipt.receipt_doc->'parent_receipt_hashes'
              ) IS DISTINCT FROM 2
           OR NOT (
               coordinator_receipt.receipt_doc->'parent_receipt_hashes'
               @> pg_catalog.jsonb_build_array(
                   candidate_row.chain_state_receipt_hash,
                   p_last_legacy_finalization_receipt_hash
               )
           ) THEN
            RAISE EXCEPTION 'stateful epoch cutover binding coordinator receipt is invalid';
        END IF;
    END IF;

    IF state_row.mapping_hash IS NULL THEN
        UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
        SET mapping_hash = candidate_row.mapping_hash,
            candidate_snapshot_hash = candidate_row.snapshot_hash,
            candidate_receipt_hash = candidate_row.chain_state_receipt_hash,
            last_legacy_finalization_receipt_hash =
                p_last_legacy_finalization_receipt_hash,
            cutover_authority_hash = p_cutover_authority_hash,
            updated_at = pg_catalog.clock_timestamp()
        WHERE singleton
        RETURNING * INTO state_row;
    ELSIF state_row.mapping_hash IS DISTINCT FROM p_mapping_hash
       OR state_row.candidate_snapshot_hash IS DISTINCT FROM candidate_row.snapshot_hash
       OR state_row.candidate_receipt_hash IS DISTINCT FROM
          candidate_row.chain_state_receipt_hash
       OR state_row.last_legacy_finalization_receipt_hash IS DISTINCT FROM
          p_last_legacy_finalization_receipt_hash
       OR state_row.cutover_authority_hash IS DISTINCT FROM p_cutover_authority_hash THEN
        RAISE EXCEPTION 'stateful epoch cutover binding conflicts with durable plan';
    END IF;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.mapping_hash,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.last_legacy_epoch_id::BIGINT,
        state_row.first_settlement_epoch_id::BIGINT,
        state_row.candidate_snapshot_hash,
        state_row.candidate_receipt_hash,
        state_row.cutover_authority_hash,
        state_row.cutover_receipt_hash,
        state_row.initialization_nonce,
        state_row.initialization_payload_hash;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_bind_v1(TEXT, TEXT, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_bind_v1(TEXT, TEXT, TEXT, TEXT)
    TO service_role;

-- Every currently deployed physical epoch identity is guarded by this one
-- trigger while the fence is closed.  The table-specific branches are the
-- complete exception list: the already-bound candidate, one measured
-- coordinator receipt, the exact cutover row inside the atomic RPC, and the
-- exact first EPOCH_INITIALIZATION row.  All normal stateful writes remain
-- blocked in stateful_staged until the explicit activation RPC is called.
CREATE OR REPLACE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    row_doc JSONB := pg_catalog.to_jsonb(NEW);
    identity_key TEXT;
    identity_text TEXT;
    identity_value BIGINT;
    payload JSONB;
    authority_value JSONB;
    has_stateful_receipt BOOLEAN;
    linked_bundle RECORD;
BEGIN
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'stateful epoch cutover singleton state is missing';
    END IF;
    IF state_row.lifecycle_state IN ('cutover_fenced', 'stateful_staged') THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
        SELECT * INTO state_row
        FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
        WHERE singleton;
    END IF;

    IF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_candidates_v1' THEN
        IF state_row.lifecycle_state = 'legacy_open' THEN
            RAISE EXCEPTION
                'stateful epoch candidate requires the durable pre-boundary fence';
        END IF;
        IF state_row.lifecycle_state = 'cutover_fenced'
           AND state_row.mapping_hash IS NULL
           AND row_doc->>'network_genesis_hash' = state_row.network_genesis_hash
           AND (row_doc->>'netuid')::INTEGER = state_row.netuid
           AND (row_doc->>'proposed_settlement_epoch_id')::INTEGER =
               state_row.first_settlement_epoch_id THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'mapping_hash' IS NOT DISTINCT FROM state_row.mapping_hash
           AND row_doc->>'snapshot_hash' IS NOT DISTINCT FROM
              state_row.candidate_snapshot_hash
           AND row_doc->>'chain_state_receipt_hash' IS NOT DISTINCT FROM
              state_row.candidate_receipt_hash THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects an unbound candidate';
    ELSIF TG_TABLE_NAME = 'research_lab_stateful_subnet_epoch_cutovers_v1' THEN
        IF state_row.cutover_receipt_hash IS NOT NULL
           AND state_row.initialization_nonce IS NOT NULL
           AND state_row.initialization_payload_hash IS NOT NULL
           AND row_doc->>'mapping_hash' IS NOT DISTINCT FROM state_row.mapping_hash
           AND row_doc->>'cutover_authority_hash' IS NOT DISTINCT FROM
              state_row.cutover_authority_hash
           AND row_doc->>'cutover_receipt_hash' IS NOT DISTINCT FROM
              state_row.cutover_receipt_hash
           AND (row_doc->>'first_settlement_epoch_id')::BIGINT IS NOT DISTINCT FROM
              state_row.first_settlement_epoch_id::BIGINT THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence requires the atomic staged cutover RPC';
    ELSIF TG_TABLE_NAME IN (
        'research_lab_stateful_subnet_epoch_boundaries_v1',
        'research_lab_stateful_subnet_epoch_snapshots_v1'
    ) THEN
        IF state_row.lifecycle_state = 'stateful_active' THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects boundary/snapshot writes before activation';
    END IF;

    IF state_row.lifecycle_state = 'legacy_open' THEN
        RETURN NEW;
    END IF;

    IF state_row.lifecycle_state = 'stateful_active' THEN
        -- Any document that elects to carry stateful authority must carry only
        -- the active mapping.  This is enforceable independent of the calling
        -- runtime and catches stale/mixed authority envelopes at the database
        -- boundary.
        FOR authority_value IN
            SELECT value
            FROM pg_catalog.jsonb_path_query(
                row_doc,
                'strict $.**.cutover_mapping_hash'
            ) AS authority_item(value)
        LOOP
            IF pg_catalog.jsonb_typeof(authority_value) <> 'string'
               OR authority_value #>> '{}' <> state_row.mapping_hash THEN
                RAISE EXCEPTION
                    'stateful epoch active mapping authority differs on %',
                    TG_TABLE_NAME;
            END IF;
        END LOOP;

        IF TG_TABLE_NAME = 'transparency_log' THEN
            payload := row_doc->'payload';
            IF row_doc ? 'epoch_id'
               AND pg_catalog.jsonb_typeof(row_doc->'epoch_id') <> 'null'
               AND (
                   row_doc->>'epoch_id' !~ '^[0-9]+$'
                   OR (row_doc->>'epoch_id')::BIGINT >=
                      state_row.first_settlement_epoch_id::BIGINT
               ) THEN
                RAISE EXCEPTION
                    'stateful epoch active transparency physical identity is forbidden';
            END IF;
            IF pg_catalog.jsonb_typeof(payload) <> 'object'
               OR NOT (payload ? 'epoch_id') THEN
                RETURN NEW;
            END IF;
            identity_text := payload->>'epoch_id';
            IF identity_text !~ '^[0-9]+$' THEN
                RAISE EXCEPTION
                    'stateful epoch fence rejects malformed transparency epoch identity';
            END IF;
            identity_value := identity_text::BIGINT;
            IF identity_value >= state_row.first_settlement_epoch_id::BIGINT
               AND row_doc->>'event_type' IN (
                   'EPOCH_INITIALIZATION', 'EPOCH_END', 'EPOCH_INPUTS'
               )
               AND (
                   payload->>'epoch_key_semantics' IS DISTINCT FROM
                      'settlement_ordinal'
                   OR payload->'epoch_authority'->>'cutover_mapping_hash'
                      IS DISTINCT FROM state_row.mapping_hash
               ) THEN
                RAISE EXCEPTION
                    'stateful epoch active lifecycle identity lacks exact authority %',
                    identity_value;
            END IF;
            RETURN NEW;
        END IF;

        -- V1/legacy weight authorities are retired at activation. They cannot
        -- establish an official stateful mapping, so a stale process must not
        -- extend them into the reserved settlement namespace.
        IF TG_TABLE_NAME IN (
            'published_weight_bundles',
            'research_lab_attested_execution_receipts',
            'research_lab_attested_weight_bundles',
            'research_lab_legacy_finalized_allocation_migrations_v2'
        ) THEN
            FOREACH identity_key IN ARRAY ARRAY[
                'epoch', 'epoch_id', 'evaluation_epoch'
            ]
            LOOP
                IF row_doc ? identity_key
                   AND pg_catalog.jsonb_typeof(row_doc->identity_key) <> 'null'
                   AND row_doc->>identity_key ~ '^[0-9]+$'
                   AND (row_doc->>identity_key)::BIGINT >=
                       state_row.first_settlement_epoch_id::BIGINT THEN
                    RAISE EXCEPTION
                        'stateful epoch active namespace rejects legacy %.% identity %',
                        TG_TABLE_NAME,
                        identity_key,
                        row_doc->>identity_key;
                END IF;
            END LOOP;
            RETURN NEW;
        END IF;

        -- A stateful validator graph persists its subnet snapshot receipt
        -- parent-first. Critical V2 weight receipts may therefore proceed only
        -- after that same-epoch receipt exists. Gateway input receipts are not
        -- constrained here because they are intentionally produced before the
        -- validator reads finalized chain state.
        IF TG_TABLE_NAME = 'research_lab_attested_execution_receipts_v2'
           AND row_doc->>'epoch_id' ~ '^[0-9]+$'
           AND (row_doc->>'epoch_id')::BIGINT >=
               state_row.first_settlement_epoch_id::BIGINT
           AND row_doc->>'purpose' IN (
               'validator.weight_snapshot.v2',
               'validator.weights.computed.v2',
               'validator.weights.finalized.v2',
               'gateway.weights.publication.v2'
           )
           AND NOT EXISTS (
               SELECT 1
               FROM public.research_lab_attested_execution_receipts_v2 AS receipt
               WHERE receipt.epoch_id = (row_doc->>'epoch_id')::INTEGER
                 AND receipt.role = 'validator_weights'
                 AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                 AND receipt.receipt_status = 'succeeded'
           ) THEN
            RAISE EXCEPTION
                'stateful epoch active V2 receipt lacks subnet authority %',
                row_doc->>'epoch_id';
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_weight_bundles_v2'
           AND row_doc->>'epoch_id' ~ '^[0-9]+$'
           AND (row_doc->>'epoch_id')::BIGINT >=
               state_row.first_settlement_epoch_id::BIGINT THEN
            payload := row_doc->'bundle_doc'->'receipt_graph'->'receipts';
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.jsonb_array_elements(payload) AS graph_receipt(doc)
                JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
                  ON receipt.receipt_hash = graph_receipt.doc->>'receipt_hash'
                WHERE graph_receipt.doc->>'role' = 'validator_weights'
                  AND graph_receipt.doc->>'purpose' =
                      'validator.subnet_epoch_snapshot.v2'
                  AND graph_receipt.doc->>'epoch_id' ~ '^[0-9]+$'
                  AND (graph_receipt.doc->>'epoch_id')::BIGINT =
                      (row_doc->>'epoch_id')::BIGINT
                  AND receipt.epoch_id = (row_doc->>'epoch_id')::INTEGER
                  AND receipt.role = 'validator_weights'
                  AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                  AND receipt.receipt_status = 'succeeded'
            ) INTO has_stateful_receipt
            WHERE pg_catalog.jsonb_typeof(payload) = 'array';
            IF NOT COALESCE(has_stateful_receipt, FALSE) THEN
                RAISE EXCEPTION
                    'stateful epoch active V2 bundle lacks subnet authority %',
                    row_doc->>'epoch_id';
            END IF;
        END IF;

        IF TG_TABLE_NAME = 'research_lab_attested_weight_finalizations_v2' THEN
            SELECT bundle.epoch_id, bundle.bundle_doc
            INTO linked_bundle
            FROM public.research_lab_attested_weight_bundles_v2 AS bundle
            WHERE bundle.bundle_hash = row_doc->>'bundle_hash';
            IF FOUND
               AND linked_bundle.epoch_id >=
                   state_row.first_settlement_epoch_id THEN
                payload := linked_bundle.bundle_doc->'receipt_graph'->'receipts';
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.jsonb_array_elements(payload)
                         AS graph_receipt(doc)
                    JOIN public.research_lab_attested_execution_receipts_v2 AS receipt
                      ON receipt.receipt_hash = graph_receipt.doc->>'receipt_hash'
                    WHERE graph_receipt.doc->>'purpose' =
                          'validator.subnet_epoch_snapshot.v2'
                      AND graph_receipt.doc->>'epoch_id' ~ '^[0-9]+$'
                      AND (graph_receipt.doc->>'epoch_id')::INTEGER =
                          linked_bundle.epoch_id
                      AND receipt.epoch_id = linked_bundle.epoch_id
                      AND receipt.purpose = 'validator.subnet_epoch_snapshot.v2'
                      AND receipt.receipt_status = 'succeeded'
                ) INTO has_stateful_receipt
                WHERE pg_catalog.jsonb_typeof(payload) = 'array';
                IF NOT COALESCE(has_stateful_receipt, FALSE) THEN
                    RAISE EXCEPTION
                        'stateful epoch active V2 finalization lacks subnet authority %',
                        linked_bundle.epoch_id;
                END IF;
            END IF;
        END IF;

        -- Rows in older generic business tables do not all carry an authority
        -- document. Their residual active-mode protection is the shared
        -- gateway epoch gate; the database still enforces every mapping or
        -- critical weight/lifecycle invariant represented in the row itself.
        RETURN NEW;
    END IF;

    IF TG_TABLE_NAME = 'research_lab_attested_execution_receipts_v2'
       AND row_doc ? 'epoch_id'
       AND row_doc->>'epoch_id' ~ '^[0-9]+$'
       AND (row_doc->>'epoch_id')::BIGINT >=
           state_row.first_settlement_epoch_id::BIGINT THEN
        IF row_doc->>'role' = 'validator_weights'
           AND row_doc->>'purpose' = 'validator.subnet_epoch_snapshot.v2'
           AND row_doc->>'receipt_status' = 'succeeded'
           AND row_doc->>'output_root' ~ '^sha256:[0-9a-f]{64}$'
           AND (row_doc->>'epoch_id')::BIGINT =
               state_row.first_settlement_epoch_id::BIGINT
           AND (
               state_row.candidate_receipt_hash IS NULL
               OR (
                   row_doc->>'receipt_hash' = state_row.candidate_receipt_hash
                   AND row_doc->>'output_root' = state_row.candidate_snapshot_hash
               )
           ) THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'role' = 'gateway_coordinator'
           AND row_doc->>'purpose' = 'research_lab.subnet_epoch_cutover.v2'
           AND row_doc->>'receipt_status' = 'succeeded'
           AND row_doc->>'output_root' = state_row.cutover_authority_hash
           AND (row_doc->>'epoch_id')::BIGINT =
               state_row.first_settlement_epoch_id::BIGINT
           AND pg_catalog.jsonb_typeof(
                  row_doc->'receipt_doc'->'parent_receipt_hashes'
               ) = 'array'
           AND pg_catalog.jsonb_array_length(
                  row_doc->'receipt_doc'->'parent_receipt_hashes'
               ) = 2
           AND row_doc->'receipt_doc'->'parent_receipt_hashes'
               @> pg_catalog.jsonb_build_array(
                   state_row.candidate_receipt_hash,
                   state_row.last_legacy_finalization_receipt_hash
               ) THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects receipt epoch identity %',
            row_doc->>'epoch_id';
    END IF;

    IF TG_TABLE_NAME = 'transparency_log' THEN
        payload := row_doc->'payload';
        IF row_doc ? 'epoch_id'
           AND pg_catalog.jsonb_typeof(row_doc->'epoch_id') <> 'null'
           AND (
               row_doc->>'epoch_id' !~ '^[0-9]+$'
               OR (row_doc->>'epoch_id')::BIGINT >=
                  state_row.first_settlement_epoch_id::BIGINT
           ) THEN
            RAISE EXCEPTION
                'stateful epoch fence rejects transparency physical identity';
        END IF;
        IF pg_catalog.jsonb_typeof(payload) <> 'object'
           OR NOT (payload ? 'epoch_id') THEN
            RETURN NEW;
        END IF;
        identity_text := payload->>'epoch_id';
        IF identity_text !~ '^[0-9]+$' THEN
            RAISE EXCEPTION 'stateful epoch fence rejects malformed transparency epoch identity';
        END IF;
        identity_value := identity_text::BIGINT;
        IF identity_value < state_row.first_settlement_epoch_id::BIGINT THEN
            RETURN NEW;
        END IF;
        IF row_doc->>'event_type' = 'EPOCH_INITIALIZATION'
           AND identity_value = state_row.first_settlement_epoch_id::BIGINT
           AND payload->>'epoch_key_semantics' = 'settlement_ordinal'
           AND payload->'epoch_authority'->>'cutover_mapping_hash' =
               state_row.mapping_hash
           AND row_doc->>'nonce' = state_row.initialization_nonce::TEXT
           AND row_doc->>'payload_hash' = state_row.initialization_payload_hash THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'stateful epoch fence rejects transparency epoch identity %',
            identity_value;
    END IF;

    FOREACH identity_key IN ARRAY ARRAY['epoch', 'epoch_id', 'evaluation_epoch']
    LOOP
        IF NOT (row_doc ? identity_key)
           OR pg_catalog.jsonb_typeof(row_doc->identity_key) = 'null' THEN
            CONTINUE;
        END IF;
        identity_text := row_doc->>identity_key;
        IF identity_text !~ '^-?[0-9]+$' THEN
            RAISE EXCEPTION 'stateful epoch fence rejects malformed %.% identity',
                TG_TABLE_NAME, identity_key;
        END IF;
        identity_value := identity_text::BIGINT;
        IF identity_value >= state_row.first_settlement_epoch_id::BIGINT THEN
            RAISE EXCEPTION 'stateful epoch fence rejects %.% identity %',
                TG_TABLE_NAME, identity_key, identity_value;
        END IF;
    END LOOP;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()
    FROM PUBLIC, anon, authenticated;

-- Commit the measured cutover and the exact current-shape legacy lifecycle
-- row in one transaction.  The payload is intentionally supplied by the
-- gateway's existing lifecycle builder; SQL validates and binds the exact
-- payload/hash/nonce rather than inventing a second event format.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_stage_v1(
    p_cutover_row JSONB,
    p_initialization_event JSONB
)
RETURNS TABLE (
    lifecycle_state                 TEXT,
    mapping_hash                    TEXT,
    cutover_authority_hash          TEXT,
    cutover_receipt_hash            TEXT,
    initialization_nonce            UUID,
    initialization_payload_hash     TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    cutover_input public.research_lab_stateful_subnet_epoch_cutovers_v1%ROWTYPE;
    stored_cutover public.research_lab_stateful_subnet_epoch_cutovers_v1%ROWTYPE;
    initialization_row RECORD;
    initialization_payload JSONB;
    initialization_nonce_value UUID;
    initialization_ts TIMESTAMPTZ;
BEGIN
    IF pg_catalog.jsonb_typeof(p_cutover_row) IS DISTINCT FROM 'object'
       OR NOT (p_cutover_row ?& ARRAY[
           'cutover_authority_hash', 'schema_version', 'mapping_hash',
           'manifest_schema_version', 'epoch_scheme', 'previous_epoch_scheme',
           'network_genesis_hash', 'netuid', 'cutover_block',
           'cutover_block_hash', 'first_subnet_epoch_index', 'first_epoch_ref',
           'first_settlement_epoch_id', 'last_legacy_epoch_id', 'first_tempo',
           'first_pending_epoch_at', 'first_blocks_since_last_step',
           'first_next_epoch_block', 'first_observed_at', 'first_snapshot_hash',
           'first_snapshot_receipt_hash', 'last_legacy_bundle_hash',
           'last_legacy_weight_finalization_event_hash',
           'last_legacy_finalization_receipt_hash', 'cutover_receipt_hash',
           'manifest_doc', 'first_snapshot_doc', 'authority_doc'
       ])
       OR p_cutover_row - ARRAY[
           'cutover_authority_hash', 'schema_version', 'mapping_hash',
           'manifest_schema_version', 'epoch_scheme', 'previous_epoch_scheme',
           'network_genesis_hash', 'netuid', 'cutover_block',
           'cutover_block_hash', 'first_subnet_epoch_index', 'first_epoch_ref',
           'first_settlement_epoch_id', 'last_legacy_epoch_id', 'first_tempo',
           'first_pending_epoch_at', 'first_blocks_since_last_step',
           'first_next_epoch_block', 'first_observed_at', 'first_snapshot_hash',
           'first_snapshot_receipt_hash', 'last_legacy_bundle_hash',
           'last_legacy_weight_finalization_event_hash',
           'last_legacy_finalization_receipt_hash', 'cutover_receipt_hash',
           'manifest_doc', 'first_snapshot_doc', 'authority_doc'
       ] <> '{}'::JSONB THEN
        RAISE EXCEPTION 'stateful epoch staged cutover row shape is invalid';
    END IF;
    IF pg_catalog.jsonb_typeof(p_initialization_event) IS DISTINCT FROM 'object'
       OR NOT (p_initialization_event ?& ARRAY[
           'event_type', 'actor_hotkey', 'nonce', 'ts', 'payload_hash',
           'build_id', 'signature', 'payload'
       ])
       OR p_initialization_event - ARRAY[
           'event_type', 'actor_hotkey', 'nonce', 'ts', 'payload_hash',
           'build_id', 'signature', 'payload'
       ] <> '{}'::JSONB THEN
        RAISE EXCEPTION 'stateful epoch initialization event shape is invalid';
    END IF;

    SELECT * INTO cutover_input
    FROM pg_catalog.jsonb_populate_record(
        NULL::public.research_lab_stateful_subnet_epoch_cutovers_v1,
        p_cutover_row
    );
    initialization_payload := p_initialization_event->'payload';
    initialization_nonce_value := (p_initialization_event->>'nonce')::UUID;
    initialization_ts := (p_initialization_event->>'ts')::TIMESTAMPTZ;

    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND
       OR state_row.lifecycle_state NOT IN (
           'cutover_fenced', 'stateful_staged', 'stateful_active'
       ) THEN
        RAISE EXCEPTION 'stateful epoch cutover must be fenced before staging';
    END IF;
    IF cutover_input.mapping_hash IS DISTINCT FROM state_row.mapping_hash
       OR cutover_input.cutover_authority_hash IS DISTINCT FROM
          state_row.cutover_authority_hash
       OR cutover_input.first_settlement_epoch_id IS DISTINCT FROM
          state_row.first_settlement_epoch_id
       OR cutover_input.last_legacy_epoch_id IS DISTINCT FROM
          state_row.last_legacy_epoch_id
       OR cutover_input.first_snapshot_hash IS DISTINCT FROM
          state_row.candidate_snapshot_hash
       OR cutover_input.first_snapshot_receipt_hash IS DISTINCT FROM
          state_row.candidate_receipt_hash
       OR cutover_input.last_legacy_finalization_receipt_hash IS DISTINCT FROM
          state_row.last_legacy_finalization_receipt_hash THEN
        RAISE EXCEPTION 'stateful epoch staged cutover differs from fenced plan';
    END IF;
    IF p_initialization_event->>'event_type' <> 'EPOCH_INITIALIZATION'
       OR p_initialization_event->>'actor_hotkey' <> 'system'
       OR p_initialization_event->>'signature' <> 'system'
       OR p_initialization_event->>'payload_hash' !~ '^[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(initialization_payload) IS DISTINCT FROM 'object'
       OR (initialization_payload->>'epoch_id')::BIGINT IS DISTINCT FROM
          state_row.first_settlement_epoch_id::BIGINT
       OR initialization_payload->>'epoch_key_semantics' <>
          'settlement_ordinal'
       OR initialization_payload->'epoch_authority' IS DISTINCT FROM
          cutover_input.first_snapshot_doc
       OR (initialization_payload->'epoch_boundaries'->>'start_block')::BIGINT
          IS DISTINCT FROM cutover_input.cutover_block
       OR (initialization_payload->'epoch_boundaries'->>'end_block')::BIGINT
          IS DISTINCT FROM cutover_input.first_next_epoch_block
       OR (initialization_payload->'epoch_boundaries'->>'expected_end_block')::BIGINT
          IS DISTINCT FROM cutover_input.first_next_epoch_block
       OR (initialization_payload->'epoch_boundaries'->>'pending_epoch_at')::BIGINT
          IS DISTINCT FROM cutover_input.first_pending_epoch_at
       OR (initialization_payload->'epoch_boundaries'->>'tempo')::INTEGER
          IS DISTINCT FROM cutover_input.first_tempo THEN
        RAISE EXCEPTION 'stateful epoch initialization differs from cutover authority';
    END IF;

    IF state_row.lifecycle_state IN ('stateful_staged', 'stateful_active')
       AND (
           state_row.cutover_receipt_hash IS DISTINCT FROM
              cutover_input.cutover_receipt_hash
           OR state_row.initialization_nonce IS DISTINCT FROM
              initialization_nonce_value
           OR state_row.initialization_payload_hash IS DISTINCT FROM
              p_initialization_event->>'payload_hash'
       ) THEN
        RAISE EXCEPTION 'stateful epoch staged retry conflicts with durable plan';
    END IF;

    UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    SET cutover_receipt_hash = cutover_input.cutover_receipt_hash,
        initialization_nonce = initialization_nonce_value,
        initialization_payload_hash = p_initialization_event->>'payload_hash',
        updated_at = pg_catalog.clock_timestamp()
    WHERE singleton
    RETURNING * INTO state_row;

    INSERT INTO public.research_lab_stateful_subnet_epoch_cutovers_v1 (
        cutover_authority_hash, schema_version, mapping_hash,
        manifest_schema_version, epoch_scheme, previous_epoch_scheme,
        network_genesis_hash, netuid, cutover_block, cutover_block_hash,
        first_subnet_epoch_index, first_epoch_ref, first_settlement_epoch_id,
        last_legacy_epoch_id, first_tempo, first_pending_epoch_at,
        first_blocks_since_last_step, first_next_epoch_block, first_observed_at,
        first_snapshot_hash, first_snapshot_receipt_hash,
        last_legacy_bundle_hash, last_legacy_weight_finalization_event_hash,
        last_legacy_finalization_receipt_hash, cutover_receipt_hash,
        manifest_doc, first_snapshot_doc, authority_doc
    ) VALUES (
        cutover_input.cutover_authority_hash, cutover_input.schema_version,
        cutover_input.mapping_hash, cutover_input.manifest_schema_version,
        cutover_input.epoch_scheme, cutover_input.previous_epoch_scheme,
        cutover_input.network_genesis_hash, cutover_input.netuid,
        cutover_input.cutover_block, cutover_input.cutover_block_hash,
        cutover_input.first_subnet_epoch_index, cutover_input.first_epoch_ref,
        cutover_input.first_settlement_epoch_id,
        cutover_input.last_legacy_epoch_id, cutover_input.first_tempo,
        cutover_input.first_pending_epoch_at,
        cutover_input.first_blocks_since_last_step,
        cutover_input.first_next_epoch_block, cutover_input.first_observed_at,
        cutover_input.first_snapshot_hash,
        cutover_input.first_snapshot_receipt_hash,
        cutover_input.last_legacy_bundle_hash,
        cutover_input.last_legacy_weight_finalization_event_hash,
        cutover_input.last_legacy_finalization_receipt_hash,
        cutover_input.cutover_receipt_hash, cutover_input.manifest_doc,
        cutover_input.first_snapshot_doc, cutover_input.authority_doc
    )
    ON CONFLICT DO NOTHING;

    SELECT cutover.* INTO stored_cutover
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 AS cutover
    WHERE cutover.mapping_hash = state_row.mapping_hash;
    IF NOT FOUND
       OR (
           pg_catalog.to_jsonb(stored_cutover) - 'created_at'
           IS DISTINCT FROM
           pg_catalog.to_jsonb(cutover_input) - 'created_at'
       ) THEN
        RAISE EXCEPTION 'stateful epoch staged cutover exact readback failed';
    END IF;

    INSERT INTO public.transparency_log (
        event_type, actor_hotkey, nonce, ts, payload_hash, build_id,
        signature, payload
    ) VALUES (
        p_initialization_event->>'event_type',
        p_initialization_event->>'actor_hotkey',
        initialization_nonce_value,
        initialization_ts,
        p_initialization_event->>'payload_hash',
        p_initialization_event->>'build_id',
        p_initialization_event->>'signature',
        initialization_payload
    )
    ON CONFLICT DO NOTHING;

    SELECT id, event_type, actor_hotkey, nonce, ts, payload_hash, build_id,
           signature, payload
    INTO initialization_row
    FROM public.transparency_log
    WHERE nonce = initialization_nonce_value;
    IF NOT FOUND
       OR initialization_row.event_type IS DISTINCT FROM
          p_initialization_event->>'event_type'
       OR initialization_row.actor_hotkey IS DISTINCT FROM
          p_initialization_event->>'actor_hotkey'
       OR initialization_row.nonce IS DISTINCT FROM initialization_nonce_value
       OR initialization_row.ts IS DISTINCT FROM initialization_ts
       OR initialization_row.payload_hash IS DISTINCT FROM
          p_initialization_event->>'payload_hash'
       OR initialization_row.build_id IS DISTINCT FROM
          p_initialization_event->>'build_id'
       OR initialization_row.signature IS DISTINCT FROM
          p_initialization_event->>'signature'
       OR initialization_row.payload IS DISTINCT FROM initialization_payload THEN
        RAISE EXCEPTION 'stateful epoch initialization exact readback failed';
    END IF;

    UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    SET lifecycle_state = 'stateful_staged',
        staged_at = COALESCE(staged_at, pg_catalog.clock_timestamp()),
        updated_at = pg_catalog.clock_timestamp()
    WHERE singleton
    RETURNING * INTO state_row;

    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.mapping_hash,
        state_row.cutover_authority_hash,
        state_row.cutover_receipt_hash,
        state_row.initialization_nonce,
        state_row.initialization_payload_hash;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_stage_v1(JSONB, JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_stage_v1(JSONB, JSONB)
    TO service_role;

-- Deliberately separate from staging.  All legacy writers remain stopped while
-- the operator prepares the exact stateful release/env offline, calls this
-- function to open the namespace, and only then starts stateful runtimes.
-- Post-start loaded-code/log/DB/on-chain verification remains mandatory; it
-- cannot occur while the runtime authority gate correctly rejects staged mode.
CREATE OR REPLACE FUNCTION public.research_lab_stateful_subnet_epoch_activate_v1(
    p_mapping_hash TEXT,
    p_confirm_stateful_release_prepared BOOLEAN
)
RETURNS TABLE (
    lifecycle_state                 TEXT,
    mapping_hash                    TEXT,
    cutover_authority_hash          TEXT,
    cutover_receipt_hash            TEXT,
    initialization_nonce            UUID,
    initialization_payload_hash     TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    state_row public.research_lab_stateful_subnet_epoch_cutover_state_v1%ROWTYPE;
    cutover_count BIGINT;
    initialization_count BIGINT;
BEGIN
    IF p_confirm_stateful_release_prepared IS DISTINCT FROM TRUE THEN
        RAISE EXCEPTION 'stateful epoch activation requires explicit offline release preparation';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(7100, 0);
    SELECT * INTO state_row
    FROM public.research_lab_stateful_subnet_epoch_cutover_state_v1
    WHERE singleton
    FOR UPDATE;
    IF NOT FOUND
       OR state_row.mapping_hash IS DISTINCT FROM p_mapping_hash
       OR state_row.lifecycle_state NOT IN ('stateful_staged', 'stateful_active') THEN
        RAISE EXCEPTION 'stateful epoch activation has no exact staged plan';
    END IF;

    SELECT COUNT(*) INTO cutover_count
    FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 AS cutover
    WHERE cutover.mapping_hash = state_row.mapping_hash
      AND cutover.cutover_authority_hash = state_row.cutover_authority_hash
      AND cutover.cutover_receipt_hash = state_row.cutover_receipt_hash;
    SELECT COUNT(*) INTO initialization_count
    FROM public.transparency_log
    WHERE nonce = state_row.initialization_nonce
      AND event_type = 'EPOCH_INITIALIZATION'
      AND payload_hash = state_row.initialization_payload_hash
      AND payload->>'epoch_key_semantics' = 'settlement_ordinal'
      AND (payload->>'epoch_id')::BIGINT =
          state_row.first_settlement_epoch_id::BIGINT
      AND payload->'epoch_authority'->>'cutover_mapping_hash' =
          state_row.mapping_hash;
    IF cutover_count IS DISTINCT FROM 1
       OR initialization_count IS DISTINCT FROM 1 THEN
        RAISE EXCEPTION 'stateful epoch activation durable readback failed';
    END IF;

    IF state_row.lifecycle_state = 'stateful_staged' THEN
        UPDATE public.research_lab_stateful_subnet_epoch_cutover_state_v1
        SET lifecycle_state = 'stateful_active',
            activated_at = pg_catalog.clock_timestamp(),
            updated_at = pg_catalog.clock_timestamp()
        WHERE singleton
        RETURNING * INTO state_row;
    END IF;
    RETURN QUERY SELECT
        state_row.lifecycle_state,
        state_row.mapping_hash,
        state_row.cutover_authority_hash,
        state_row.cutover_receipt_hash,
        state_row.initialization_nonce,
        state_row.initialization_payload_hash;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_activate_v1(TEXT, BOOLEAN)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_stateful_subnet_epoch_activate_v1(TEXT, BOOLEAN)
    TO service_role;

DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_candidate_v1
    ON public.research_lab_stateful_subnet_epoch_candidates_v1;
CREATE TRIGGER validate_research_lab_stateful_epoch_candidate_v1
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_candidates_v1
    FOR EACH ROW EXECUTE FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1();

DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_cutover_v1
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE TRIGGER validate_research_lab_stateful_epoch_cutover_v1
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW EXECUTE FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1();

DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_boundary_v1
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1;
CREATE TRIGGER validate_research_lab_stateful_epoch_boundary_v1
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_boundaries_v1
    FOR EACH ROW EXECUTE FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1();

DROP TRIGGER IF EXISTS validate_research_lab_stateful_epoch_snapshot_v1
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1;
CREATE TRIGGER validate_research_lab_stateful_epoch_snapshot_v1
    BEFORE INSERT ON public.research_lab_stateful_subnet_epoch_snapshots_v1
    FOR EACH ROW EXECUTE FUNCTION public.validate_research_lab_stateful_subnet_epoch_v1();

-- Attach the fence to every current public table that owns a physical integer
-- epoch identity, plus the JSON epoch log and the four stateful authority
-- tables.  Partition children inherit the parent trigger and are excluded
-- from this catalog loop to avoid duplicate execution.
DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT DISTINCT c.oid, n.nspname, c.relname
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_catalog.pg_attribute a
          ON a.attrelid = c.oid
         AND a.attnum > 0
         AND NOT a.attisdropped
        WHERE n.nspname = 'public'
          AND (
              c.relkind = 'p'
              OR (c.relkind = 'r' AND NOT c.relispartition)
          )
          AND c.relname <>
              'research_lab_stateful_subnet_epoch_cutover_state_v1'
          AND (
              (
                  a.atttypid IN (20, 21, 23)
                  AND a.attname IN ('epoch', 'epoch_id', 'evaluation_epoch')
              )
              OR c.relname IN (
                  'transparency_log',
                  'research_lab_stateful_subnet_epoch_candidates_v1',
                  'research_lab_stateful_subnet_epoch_cutovers_v1',
                  'research_lab_stateful_subnet_epoch_boundaries_v1',
                  'research_lab_stateful_subnet_epoch_snapshots_v1',
                  'research_lab_attested_weight_finalizations_v2'
              )
          )
        ORDER BY c.oid
    LOOP
        EXECUTE pg_catalog.format(
            'DROP TRIGGER IF EXISTS enforce_research_lab_stateful_epoch_fence_v1 ON %I.%I',
            item.nspname,
            item.relname
        );
        EXECUTE pg_catalog.format(
            'CREATE TRIGGER enforce_research_lab_stateful_epoch_fence_v1 '
            'BEFORE INSERT OR UPDATE ON %I.%I FOR EACH ROW '
            'EXECUTE FUNCTION public.enforce_research_lab_stateful_epoch_fence_v1()',
            item.nspname,
            item.relname
        );
    END LOOP;
END;
$$;

DROP TRIGGER IF EXISTS prevent_research_lab_stateful_epoch_cutover_v1_mutation
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE TRIGGER prevent_research_lab_stateful_epoch_cutover_v1_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_stateful_epoch_candidate_v1_mutation
    ON public.research_lab_stateful_subnet_epoch_candidates_v1;
CREATE TRIGGER prevent_research_lab_stateful_epoch_candidate_v1_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_stateful_subnet_epoch_candidates_v1
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_stateful_epoch_boundary_v1_mutation
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1;
CREATE TRIGGER prevent_research_lab_stateful_epoch_boundary_v1_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_stateful_subnet_epoch_boundaries_v1
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_stateful_epoch_snapshot_v1_mutation
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1;
CREATE TRIGGER prevent_research_lab_stateful_epoch_snapshot_v1_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_stateful_subnet_epoch_snapshots_v1
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

CREATE INDEX IF NOT EXISTS idx_research_lab_stateful_epoch_boundary_settlement_v1
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1(
        mapping_hash, settlement_epoch_id DESC
    );
CREATE INDEX IF NOT EXISTS idx_research_lab_stateful_epoch_snapshot_epoch_v1
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1(
        mapping_hash, subnet_epoch_index DESC, current_block DESC
    );
CREATE INDEX IF NOT EXISTS idx_research_lab_stateful_epoch_candidate_lineage_v1
    ON public.research_lab_stateful_subnet_epoch_candidates_v1(
        network_genesis_hash, netuid, subnet_epoch_index DESC
    );

-- Lifecycle recovery and validation query these JSON keys on every restart and
-- request.  The general index keeps both legacy and stateful reads bounded;
-- the stateful-only unique index makes concurrent monitor/request races
-- impossible at the database boundary while leaving historical legacy
-- duplicates visible for explicit repair instead of blocking this migration.
CREATE INDEX IF NOT EXISTS idx_transparency_log_lifecycle_epoch_v1
    ON public.transparency_log(
        event_type,
        ((payload->>'epoch_id'))
    )
    WHERE event_type IN (
        'EPOCH_INITIALIZATION', 'EPOCH_END', 'EPOCH_INPUTS'
    );
CREATE UNIQUE INDEX IF NOT EXISTS uq_transparency_log_stateful_lifecycle_epoch_v1
    ON public.transparency_log(
        event_type,
        ((payload->>'epoch_id'))
    )
    WHERE event_type IN (
        'EPOCH_INITIALIZATION', 'EPOCH_END', 'EPOCH_INPUTS'
    )
      AND payload->>'epoch_key_semantics' = 'settlement_ordinal';

CREATE OR REPLACE VIEW public.research_lab_stateful_subnet_epoch_mapping_v1
WITH (security_invoker = true) AS
SELECT
    cutover.mapping_hash,
    cutover.epoch_scheme,
    cutover.network_genesis_hash,
    cutover.netuid,
    cutover.first_subnet_epoch_index AS subnet_epoch_index,
    cutover.first_epoch_ref AS epoch_ref,
    cutover.first_settlement_epoch_id AS settlement_epoch_id,
    cutover.cutover_block AS last_epoch_block,
    cutover.cutover_block_hash AS boundary_block_hash,
    cutover.first_snapshot_hash AS boundary_hash,
    cutover.first_snapshot_receipt_hash AS chain_state_receipt_hash,
    TRUE AS is_cutover_boundary,
    cutover.created_at
FROM public.research_lab_stateful_subnet_epoch_cutovers_v1 cutover
UNION ALL
SELECT
    boundary.mapping_hash,
    boundary.epoch_scheme,
    boundary.network_genesis_hash,
    boundary.netuid,
    boundary.subnet_epoch_index,
    boundary.epoch_ref,
    boundary.settlement_epoch_id,
    boundary.boundary_block AS last_epoch_block,
    boundary.boundary_block_hash,
    boundary.boundary_hash,
    boundary.chain_state_receipt_hash,
    FALSE AS is_cutover_boundary,
    boundary.created_at
FROM public.research_lab_stateful_subnet_epoch_boundaries_v1 boundary;

REVOKE ALL ON TABLE public.research_lab_stateful_subnet_epoch_candidates_v1
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_stateful_subnet_epoch_boundaries_v1
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_stateful_subnet_epoch_snapshots_v1
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_stateful_subnet_epoch_mapping_v1
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()
    FROM PUBLIC, anon, authenticated, service_role;

GRANT SELECT, INSERT
    ON TABLE public.research_lab_stateful_subnet_epoch_candidates_v1
    TO service_role;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    TO service_role;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_stateful_subnet_epoch_boundaries_v1
    TO service_role;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_stateful_subnet_epoch_snapshots_v1
    TO service_role;
GRANT SELECT
    ON TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    TO service_role;
GRANT SELECT
    ON TABLE public.research_lab_stateful_subnet_epoch_mapping_v1
    TO service_role;
GRANT EXECUTE
    ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1()
    TO anon, authenticated, service_role;

ALTER TABLE public.research_lab_stateful_subnet_epoch_candidates_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_stateful_subnet_epoch_boundaries_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_stateful_subnet_epoch_snapshots_v1
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_stateful_subnet_epoch_candidates_v1;
CREATE POLICY service_role_read
    ON public.research_lab_stateful_subnet_epoch_candidates_v1
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_stateful_subnet_epoch_candidates_v1;
CREATE POLICY service_role_insert
    ON public.research_lab_stateful_subnet_epoch_candidates_v1
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE POLICY service_role_read
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1;
CREATE POLICY service_role_insert
    ON public.research_lab_stateful_subnet_epoch_cutovers_v1
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1;
CREATE POLICY service_role_read
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1;
CREATE POLICY service_role_insert
    ON public.research_lab_stateful_subnet_epoch_boundaries_v1
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1;
CREATE POLICY service_role_read
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1;
CREATE POLICY service_role_insert
    ON public.research_lab_stateful_subnet_epoch_snapshots_v1
    FOR INSERT TO service_role WITH CHECK (true);

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_stateful_subnet_epoch_cutover_state_v1;
CREATE POLICY service_role_read
    ON public.research_lab_stateful_subnet_epoch_cutover_state_v1
    FOR SELECT TO service_role USING (true);

COMMENT ON TABLE public.research_lab_stateful_subnet_epoch_candidates_v1 IS
    'Append-only validator-signed finalized boundary candidates captured while legacy mode is paused. Candidate rows never activate an epoch mapping.';
COMMENT ON TABLE public.research_lab_stateful_subnet_epoch_cutovers_v1 IS
    'Append-only, receipt-backed authority for the single collision-safe legacy_global_360_v1 to official stateful subnet epoch cutover per chain genesis/netuid. An empty table means no cutover is activated.';
COMMENT ON TABLE public.research_lab_stateful_subnet_epoch_boundaries_v1 IS
    'Append-only bijective mapping from official SubnetEpochIndex/epoch_ref boundaries to monotonic settlement_epoch_id values after the cutover.';
COMMENT ON TABLE public.research_lab_stateful_subnet_epoch_snapshots_v1 IS
    'Append-only exact-hash finalized/exact Subtensor epoch scheduler snapshots, each bound to one V2 validator.subnet_epoch_snapshot receipt and one declared stable epoch_ref mapping.';
COMMENT ON TABLE public.research_lab_stateful_subnet_epoch_cutover_state_v1 IS
    'Protected singleton cutover plan and durable write fence. Only SECURITY DEFINER fence/stage/activate RPCs may mutate it; validators read only the sanitized public-state RPC.';
COMMENT ON VIEW public.research_lab_stateful_subnet_epoch_mapping_v1 IS
    'Service-role-only canonical official epoch_ref to compatibility settlement_epoch_id mapping, including the cutover boundary exactly once.';
COMMENT ON FUNCTION public.research_lab_stateful_subnet_epoch_cutover_public_state_v1() IS
    'Anon-readable sanitized singleton lifecycle state for validator/auditor fail-closed epoch-mode refresh. Exposes no receipts, candidates, authority documents, or mutation capability.';

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Operator verification after applying (all four authority tables must initially
-- be empty; applying this migration alone does not activate stateful mode):
-- SELECT
--     (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_candidates_v1) AS candidates,
--     (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_cutovers_v1) AS cutovers,
--     (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_boundaries_v1) AS boundaries,
--     (SELECT COUNT(*) FROM public.research_lab_stateful_subnet_epoch_snapshots_v1) AS snapshots;
--
-- SELECT relname, relrowsecurity
-- FROM pg_class
-- WHERE relnamespace = 'public'::regnamespace
--   AND relname IN (
--       'research_lab_stateful_subnet_epoch_candidates_v1',
--       'research_lab_stateful_subnet_epoch_cutovers_v1',
--       'research_lab_stateful_subnet_epoch_boundaries_v1',
--       'research_lab_stateful_subnet_epoch_snapshots_v1'
--   )
-- ORDER BY relname;
--
-- SELECT mapping_hash, epoch_scheme, network_genesis_hash, netuid,
--        subnet_epoch_index, epoch_ref, settlement_epoch_id,
--        last_epoch_block, boundary_block_hash, is_cutover_boundary
-- FROM public.research_lab_stateful_subnet_epoch_mapping_v1
-- ORDER BY settlement_epoch_id;
--
-- Collision/bijection proof (must return zero rows):
-- SELECT mapping_hash, subnet_epoch_index, COUNT(DISTINCT settlement_epoch_id)
-- FROM public.research_lab_stateful_subnet_epoch_mapping_v1
-- GROUP BY mapping_hash, subnet_epoch_index
-- HAVING COUNT(DISTINCT settlement_epoch_id) <> 1
-- UNION ALL
-- SELECT mapping_hash, settlement_epoch_id, COUNT(DISTINCT subnet_epoch_index)
-- FROM public.research_lab_stateful_subnet_epoch_mapping_v1
-- GROUP BY mapping_hash, settlement_epoch_id
-- HAVING COUNT(DISTINCT subnet_epoch_index) <> 1;
--
-- Contiguous affine mapping proof (must return zero rows):
-- SELECT mapped.*
-- FROM public.research_lab_stateful_subnet_epoch_mapping_v1 mapped
-- JOIN public.research_lab_stateful_subnet_epoch_cutovers_v1 cutover
--   ON cutover.mapping_hash = mapped.mapping_hash
-- WHERE mapped.settlement_epoch_id::BIGINT <>
--       cutover.first_settlement_epoch_id::BIGINT
--       + (mapped.subnet_epoch_index - cutover.first_subnet_epoch_index);
