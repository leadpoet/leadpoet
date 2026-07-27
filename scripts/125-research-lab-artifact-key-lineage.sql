-- Scope encrypted provider state to the exact enclave artifact-key lineage.
--
-- Historical rows remain append-only and intact. Rows created before this
-- migration receive a reserved legacy lineage and are never selected by a
-- coordinator holding a different, authenticated key.

BEGIN;

ALTER TABLE public.research_lab_provider_evidence_cache_v2
    ADD COLUMN IF NOT EXISTS artifact_master_key_ref_hash TEXT;
ALTER TABLE public.research_lab_provider_evidence_cache_v2
    DROP CONSTRAINT IF EXISTS research_lab_provider_evidence_cache_v2_pkey;
CREATE UNIQUE INDEX IF NOT EXISTS
    research_lab_provider_evidence_cache_v2_key_day_request_key
    ON public.research_lab_provider_evidence_cache_v2 (
        artifact_master_key_ref_hash,
        utc_day,
        request_fingerprint
    );
CREATE UNIQUE INDEX IF NOT EXISTS
    research_lab_provider_evidence_cache_v2_legacy_day_request_key
    ON public.research_lab_provider_evidence_cache_v2 (
        utc_day,
        request_fingerprint
    )
    WHERE artifact_master_key_ref_hash IS NULL;
ALTER TABLE public.research_lab_provider_evidence_cache_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_provider_evidence_cache_v2_artifact_master_key_ref_hash_check;
ALTER TABLE public.research_lab_provider_evidence_cache_v2
    ADD CONSTRAINT
        research_lab_provider_evidence_cache_v2_artifact_master_key_ref_hash_check
    CHECK (
        artifact_master_key_ref_hash IS NULL
        OR artifact_master_key_ref_hash ~ '^sha256:[0-9a-f]{64}$'
    );

ALTER TABLE public.research_lab_provider_outcome_checkpoints_v2
    ADD COLUMN IF NOT EXISTS artifact_master_key_ref_hash TEXT;
ALTER TABLE public.research_lab_provider_outcome_checkpoints_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_provider_outcome_checkpoints_v2_utc_day_sequence_key;
CREATE UNIQUE INDEX IF NOT EXISTS
    research_lab_provider_outcome_checkpoints_v2_key_day_sequence_key
    ON public.research_lab_provider_outcome_checkpoints_v2 (
        artifact_master_key_ref_hash,
        utc_day,
        sequence
    );
CREATE UNIQUE INDEX IF NOT EXISTS
    research_lab_provider_outcome_checkpoints_v2_legacy_day_sequence_key
    ON public.research_lab_provider_outcome_checkpoints_v2 (
        utc_day,
        sequence
    )
    WHERE artifact_master_key_ref_hash IS NULL;
ALTER TABLE public.research_lab_provider_outcome_checkpoints_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_provider_outcome_checkpoints_v2_artifact_master_key_ref_hash_check;
ALTER TABLE public.research_lab_provider_outcome_checkpoints_v2
    ADD CONSTRAINT
        research_lab_provider_outcome_checkpoints_v2_artifact_master_key_ref_hash_check
    CHECK (
        artifact_master_key_ref_hash IS NULL
        OR artifact_master_key_ref_hash ~ '^sha256:[0-9a-f]{64}$'
    );

COMMIT;
