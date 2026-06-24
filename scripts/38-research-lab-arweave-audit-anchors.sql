-- Research Lab Arweave epoch audit anchors.
--
-- Deployment policy:
--   * Apply after scripts 35, 36, and 37.
--   * Stores compact Research Lab epoch audit events that are buffered into
--     the gateway TEE and later checkpointed to Arweave.
--   * No anon/authenticated grants are created.
--   * No raw OpenRouter keys, service-role keys, private repo material, hidden
--     ICP plaintext, judge prompts, private image refs, candidate patch
--     manifests, proxy credentials, or raw provider secrets may be stored here.

BEGIN;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_append_only_mutation()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = ''
AS $$
BEGIN
    RAISE EXCEPTION
        '% is append-only; write a correction or tombstone row instead',
        TG_TABLE_NAME;
END;
$$;

REVOKE ALL ON FUNCTION public.prevent_research_lab_append_only_mutation()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.prevent_research_lab_append_only_mutation()
    TO service_role;

CREATE TABLE IF NOT EXISTS public.research_lab_arweave_epoch_audit_anchors (
    anchor_id                  TEXT        PRIMARY KEY
                                            CHECK (anchor_id ~ '^research_lab_arweave_anchor:[0-9a-f]{64}$'),
    schema_version             TEXT        NOT NULL DEFAULT '1.0'
                                            CHECK (schema_version = '1.0'),
    epoch                      INTEGER     NOT NULL CHECK (epoch >= 0),
    netuid                     INTEGER     NOT NULL CHECK (netuid > 0),
    audit_kind                 TEXT        NOT NULL CHECK (audit_kind IN ('active', 'shadow')),
    audit_bundle_id            TEXT        REFERENCES public.research_lab_signed_audit_bundles(audit_bundle_id)
                                            ON DELETE RESTRICT,
    audit_bundle_hash          TEXT        CHECK (
                                            audit_bundle_hash IS NULL
                                            OR audit_bundle_hash ~ '^sha256:[0-9a-f]{64}$'
                                            ),
    allocation_hash            TEXT        CHECK (
                                            allocation_hash IS NULL
                                            OR allocation_hash ~ '^sha256:[0-9a-f]{64}$'
                                            ),
    weights_hash               TEXT,
    payload_hash               TEXT        NOT NULL CHECK (payload_hash ~ '^sha256:[0-9a-f]{64}$'),
    transparency_event_hash    TEXT        CHECK (
                                            transparency_event_hash IS NULL
                                            OR transparency_event_hash ~ '^[0-9a-f]{64}$'
                                            ),
    tee_sequence               INTEGER     CHECK (tee_sequence IS NULL OR tee_sequence >= 0),
    anchor_hash                TEXT        NOT NULL UNIQUE CHECK (anchor_hash ~ '^sha256:[0-9a-f]{64}$'),
    anchored_hash              TEXT        NOT NULL UNIQUE CHECK (anchored_hash = anchor_hash),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (epoch, netuid, audit_kind, payload_hash)
);

CREATE TABLE IF NOT EXISTS public.research_lab_arweave_epoch_audit_anchor_events (
    event_id                   UUID        PRIMARY KEY,
    schema_version             TEXT        NOT NULL DEFAULT '1.0'
                                            CHECK (schema_version = '1.0'),
    anchor_id                  TEXT        NOT NULL
                                            REFERENCES public.research_lab_arweave_epoch_audit_anchors(anchor_id)
                                            ON DELETE RESTRICT,
    seq                        INTEGER     NOT NULL CHECK (seq >= 0),
    event_type                 TEXT        NOT NULL CHECK (
                                            event_type IN (
                                                'created',
                                                'buffered',
                                                'checkpointed',
                                                'failed',
                                                'tombstoned'
                                            )),
    anchor_status              TEXT        NOT NULL CHECK (
                                            anchor_status IN (
                                                'created',
                                                'buffered',
                                                'checkpointed',
                                                'failed',
                                                'tombstoned'
                                            )),
    transparency_event_hash    TEXT        CHECK (
                                            transparency_event_hash IS NULL
                                            OR transparency_event_hash ~ '^[0-9a-f]{64}$'
                                            ),
    tee_sequence               INTEGER     CHECK (tee_sequence IS NULL OR tee_sequence >= 0),
    checkpoint_number          INTEGER     CHECK (checkpoint_number IS NULL OR checkpoint_number >= 0),
    checkpoint_merkle_root     TEXT        CHECK (
                                            checkpoint_merkle_root IS NULL
                                            OR checkpoint_merkle_root ~ '^[0-9a-f]{64}$'
                                            ),
    arweave_tx_id              TEXT,
    event_doc                  JSONB       NOT NULL DEFAULT '{}'::JSONB CHECK (
                                            jsonb_typeof(event_doc) = 'object'
                                            AND event_doc::TEXT !~* '(sk-or-|openrouter_api_key|raw_openrouter_key|raw_secret|service_role|private_repo|judge_prompt|hidden_icp|icp_plaintext|\\.dkr\\.ecr\\.|image_digest|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
                                            ),
    anchored_hash              TEXT        NOT NULL UNIQUE CHECK (anchored_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_lab_arweave_epoch_audit_anchor_events_seq_key UNIQUE (anchor_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_arweave_epoch_audit_epoch
    ON public.research_lab_arweave_epoch_audit_anchors(epoch, netuid, audit_kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_arweave_epoch_audit_event_hash
    ON public.research_lab_arweave_epoch_audit_anchors(transparency_event_hash);
CREATE INDEX IF NOT EXISTS idx_research_lab_arweave_epoch_audit_events_latest
    ON public.research_lab_arweave_epoch_audit_anchor_events(anchor_id, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_arweave_epoch_audit_events_tx
    ON public.research_lab_arweave_epoch_audit_anchor_events(arweave_tx_id);

DROP TRIGGER IF EXISTS prevent_research_lab_arweave_epoch_audit_anchors_mutation
    ON public.research_lab_arweave_epoch_audit_anchors;
CREATE TRIGGER prevent_research_lab_arweave_epoch_audit_anchors_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_arweave_epoch_audit_anchors
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

DROP TRIGGER IF EXISTS prevent_research_lab_arweave_epoch_audit_anchor_events_mutation
    ON public.research_lab_arweave_epoch_audit_anchor_events;
CREATE TRIGGER prevent_research_lab_arweave_epoch_audit_anchor_events_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_arweave_epoch_audit_anchor_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_append_only_mutation();

REVOKE ALL ON TABLE public.research_lab_arweave_epoch_audit_anchors FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_arweave_epoch_audit_anchor_events FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_arweave_epoch_audit_anchors TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_arweave_epoch_audit_anchor_events TO service_role;

ALTER TABLE public.research_lab_arweave_epoch_audit_anchors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_arweave_epoch_audit_anchor_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read ON public.research_lab_arweave_epoch_audit_anchors;
CREATE POLICY service_role_read ON public.research_lab_arweave_epoch_audit_anchors
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_arweave_epoch_audit_anchors;
CREATE POLICY service_role_insert ON public.research_lab_arweave_epoch_audit_anchors
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

DROP POLICY IF EXISTS service_role_read ON public.research_lab_arweave_epoch_audit_anchor_events;
CREATE POLICY service_role_read ON public.research_lab_arweave_epoch_audit_anchor_events
    FOR SELECT USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS service_role_insert ON public.research_lab_arweave_epoch_audit_anchor_events;
CREATE POLICY service_role_insert ON public.research_lab_arweave_epoch_audit_anchor_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE OR REPLACE VIEW public.research_lab_arweave_epoch_audit_anchor_current
WITH (security_invoker = true) AS
SELECT
    a.*,
    e.seq AS current_event_seq,
    e.event_type AS current_event_type,
    e.anchor_status AS current_anchor_status,
    e.transparency_event_hash AS current_transparency_event_hash,
    e.tee_sequence AS current_tee_sequence,
    e.checkpoint_number AS current_checkpoint_number,
    e.checkpoint_merkle_root AS current_checkpoint_merkle_root,
    e.arweave_tx_id AS current_arweave_tx_id,
    e.anchored_hash AS current_event_hash,
    e.created_at AS current_status_at
FROM public.research_lab_arweave_epoch_audit_anchors a
LEFT JOIN LATERAL (
    SELECT *
    FROM public.research_lab_arweave_epoch_audit_anchor_events e
    WHERE e.anchor_id = a.anchor_id
    ORDER BY e.seq DESC, e.created_at DESC
    LIMIT 1
) e ON TRUE;

REVOKE ALL ON TABLE public.research_lab_arweave_epoch_audit_anchor_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_arweave_epoch_audit_anchor_current TO service_role;

COMMENT ON TABLE public.research_lab_arweave_epoch_audit_anchors IS
    'Append-only Research Lab compact epoch audit anchors destined for gateway TEE Arweave checkpoints.';
COMMENT ON TABLE public.research_lab_arweave_epoch_audit_anchor_events IS
    'Append-only Research Lab Arweave audit anchor lifecycle events, including checkpoint tx linkage.';
COMMENT ON VIEW public.research_lab_arweave_epoch_audit_anchor_current IS
    'Latest Research Lab Arweave audit anchor status by anchor.';

COMMIT;
