-- Durable, service-only precompute evidence for one V2 weight submission.
--
-- This is storage schema V3 only.  It does not replace or reinterpret the
-- externally verified leadpoet.published_weight_bundle.v2 protocol.  A run
-- first binds one exact release, then one complete immutable input set, then
-- append-only stage evidence that is tied to that input set.  The final
-- readback returns precisely those stored rows in sequence order.

BEGIN;

CREATE OR REPLACE FUNCTION public.research_lab_weight_precompute_sha256_map_v3(
    p_value JSONB
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = pg_catalog
AS $$
    SELECT pg_catalog.jsonb_typeof(p_value) = 'object'
       AND p_value <> '{}'::JSONB
       AND NOT EXISTS (
           SELECT 1
           FROM pg_catalog.jsonb_each_text(p_value) AS entry(key, value)
           WHERE entry.key !~ '^[a-z][a-z0-9_.-]{0,127}$'
              OR entry.value !~ '^sha256:[0-9a-f]{64}$'
       )
$$;

CREATE OR REPLACE FUNCTION public.research_lab_weight_precompute_complete_input_set_v3(
    p_value JSONB
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
SECURITY INVOKER
SET search_path = pg_catalog
AS $$
    SELECT public.research_lab_weight_precompute_sha256_map_v3(p_value)
       AND p_value ?& ARRAY[
           'research_lab_allocation', 'champions', 'reimbursements',
           'source_add_rewards', 'fulfillment_rewards', 'leaderboard', 'bans',
           'sourcing_history', 'anomaly_adjustments'
       ]
       AND p_value - ARRAY[
           'research_lab_allocation', 'champions', 'reimbursements',
           'source_add_rewards', 'fulfillment_rewards', 'leaderboard', 'bans',
           'sourcing_history', 'anomaly_adjustments'
       ] = '{}'::JSONB
$$;

CREATE TABLE IF NOT EXISTS public.research_lab_weight_precompute_runs_v3 (
    precompute_run_id       UUID        PRIMARY KEY,
    storage_schema_version  TEXT        NOT NULL
                                        CHECK (storage_schema_version =
                                            'leadpoet.weight_precompute_run.v3'),
    protocol_schema_version TEXT        NOT NULL
                                        CHECK (protocol_schema_version =
                                            'leadpoet.published_weight_bundle.v2'),
    network_genesis_hash    TEXT        NOT NULL
                                        CHECK (network_genesis_hash ~ '^0x[0-9a-f]{64}$'),
    netuid                  INTEGER     NOT NULL CHECK (netuid > 0),
    epoch_id                BIGINT      NOT NULL CHECK (epoch_id >= 0),
    epoch_ref               TEXT        NOT NULL UNIQUE
                                        CHECK (epoch_ref ~ '^sha256:[0-9a-f]{64}$'),
    request_hash            TEXT        NOT NULL UNIQUE
                                        CHECK (request_hash ~ '^sha256:[0-9a-f]{64}$'),
    planned_submission_block BIGINT     NOT NULL CHECK (planned_submission_block >= 0),
    release_commit_sha      TEXT        NOT NULL CHECK (release_commit_sha ~ '^[0-9a-f]{40}$'),
    release_manifest_hash   TEXT        NOT NULL CHECK (release_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    run_doc                 JSONB       NOT NULL DEFAULT '{}'::JSONB
                                        CHECK (
                                            pg_catalog.jsonb_typeof(run_doc) = 'object'
                                            AND run_doc::TEXT !~* '(sk-or-|sb_secret|service_role_key|raw_secret|raw_credential|authorization|proxy-authorization|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
                                        ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (
        precompute_run_id,
        protocol_schema_version,
        release_commit_sha,
        release_manifest_hash
    ),
    UNIQUE (network_genesis_hash, netuid, epoch_id)
);

CREATE TABLE IF NOT EXISTS public.research_lab_weight_precompute_input_sets_v3 (
    precompute_run_id       UUID        PRIMARY KEY,
    storage_schema_version  TEXT        NOT NULL
                                        CHECK (storage_schema_version =
                                            'leadpoet.weight_precompute_input_set.v3'),
    protocol_schema_version TEXT        NOT NULL
                                        CHECK (protocol_schema_version =
                                            'leadpoet.published_weight_bundle.v2'),
    release_commit_sha      TEXT        NOT NULL CHECK (release_commit_sha ~ '^[0-9a-f]{40}$'),
    release_manifest_hash   TEXT        NOT NULL CHECK (release_manifest_hash ~ '^sha256:[0-9a-f]{64}$'),
    input_set_hash          TEXT        NOT NULL CHECK (input_set_hash ~ '^sha256:[0-9a-f]{64}$'),
    source_input_root       TEXT        NOT NULL CHECK (source_input_root ~ '^sha256:[0-9a-f]{64}$'),
    calculation_snapshot_hash TEXT      NOT NULL CHECK (calculation_snapshot_hash ~ '^sha256:[0-9a-f]{64}$'),
    input_receipt_hashes    JSONB       NOT NULL
                                        CHECK (public.research_lab_weight_precompute_complete_input_set_v3(input_receipt_hashes)),
    input_set_doc           JSONB       NOT NULL DEFAULT '{}'::JSONB
                                        CHECK (
                                            pg_catalog.jsonb_typeof(input_set_doc) = 'object'
                                            AND input_set_doc::TEXT !~* '(sk-or-|sb_secret|service_role_key|raw_secret|raw_credential|authorization|proxy-authorization|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
                                        ),
    completed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (precompute_run_id, input_set_hash),
    FOREIGN KEY (
        precompute_run_id,
        protocol_schema_version,
        release_commit_sha,
        release_manifest_hash
    ) REFERENCES public.research_lab_weight_precompute_runs_v3 (
        precompute_run_id,
        protocol_schema_version,
        release_commit_sha,
        release_manifest_hash
    ) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS public.research_lab_weight_precompute_stage_events_v3 (
    stage_event_id          UUID        PRIMARY KEY,
    precompute_run_id       UUID        NOT NULL,
    stage_sequence          INTEGER     NOT NULL CHECK (stage_sequence >= 0),
    storage_schema_version  TEXT        NOT NULL
                                        CHECK (storage_schema_version =
                                            'leadpoet.weight_precompute_stage_event.v3'),
    input_set_hash          TEXT        NOT NULL CHECK (input_set_hash ~ '^sha256:[0-9a-f]{64}$'),
    stage_name              TEXT        NOT NULL
                                        CHECK (stage_name ~ '^[a-z][a-z0-9_.-]{0,127}$'),
    stage_status            TEXT        NOT NULL
                                        CHECK (stage_status IN ('started', 'succeeded', 'failed')),
    event_hash              TEXT        NOT NULL UNIQUE
                                        CHECK (event_hash ~ '^sha256:[0-9a-f]{64}$'),
    event_doc               JSONB       NOT NULL DEFAULT '{}'::JSONB
                                        CHECK (
                                            pg_catalog.jsonb_typeof(event_doc) = 'object'
                                            AND event_doc::TEXT !~* '(sk-or-|sb_secret|service_role_key|raw_secret|raw_credential|authorization|proxy-authorization|provider_output|request_body|response_body|://[^/]+:[^/@]+@)'
                                        ),
    occurred_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (precompute_run_id, stage_sequence),
    FOREIGN KEY (precompute_run_id, input_set_hash)
        REFERENCES public.research_lab_weight_precompute_input_sets_v3 (
            precompute_run_id, input_set_hash
        ) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_research_lab_weight_precompute_events_v3_run_sequence
    ON public.research_lab_weight_precompute_stage_events_v3 (
        precompute_run_id, stage_sequence
    );

CREATE OR REPLACE FUNCTION public.prevent_research_lab_weight_precompute_mutation_v3()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
AS $$
BEGIN
    RAISE EXCEPTION 'research_lab_weight_precompute_append_only';
END;
$$;

DROP TRIGGER IF EXISTS prevent_research_lab_weight_precompute_runs_v3_mutation
    ON public.research_lab_weight_precompute_runs_v3;
CREATE TRIGGER prevent_research_lab_weight_precompute_runs_v3_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_weight_precompute_runs_v3
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_weight_precompute_mutation_v3();
DROP TRIGGER IF EXISTS prevent_research_lab_weight_precompute_input_sets_v3_mutation
    ON public.research_lab_weight_precompute_input_sets_v3;
CREATE TRIGGER prevent_research_lab_weight_precompute_input_sets_v3_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_weight_precompute_input_sets_v3
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_weight_precompute_mutation_v3();
DROP TRIGGER IF EXISTS prevent_research_lab_weight_precompute_stage_events_v3_mutation
    ON public.research_lab_weight_precompute_stage_events_v3;
CREATE TRIGGER prevent_research_lab_weight_precompute_stage_events_v3_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_weight_precompute_stage_events_v3
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_weight_precompute_mutation_v3();

CREATE OR REPLACE FUNCTION public.begin_research_lab_weight_precompute_run_v3(
    p_precompute_run_id UUID,
    p_network_genesis_hash TEXT,
    p_netuid INTEGER,
    p_epoch_id BIGINT,
    p_epoch_ref TEXT,
    p_request_hash TEXT,
    p_planned_submission_block BIGINT,
    p_release_commit_sha TEXT,
    p_release_manifest_hash TEXT,
    p_run_doc JSONB
)
RETURNS SETOF public.research_lab_weight_precompute_runs_v3
LANGUAGE plpgsql
VOLATILE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    existing public.research_lab_weight_precompute_runs_v3%ROWTYPE;
    inserted public.research_lab_weight_precompute_runs_v3%ROWTYPE;
BEGIN
    IF p_precompute_run_id IS NULL
       OR p_network_genesis_hash !~ '^0x[0-9a-f]{64}$'
       OR p_netuid <= 0
       OR p_epoch_id < 0
       OR p_epoch_ref !~ '^sha256:[0-9a-f]{64}$'
       OR p_request_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_planned_submission_block < 0
       OR p_release_commit_sha !~ '^[0-9a-f]{40}$'
       OR p_release_manifest_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_run_doc) IS DISTINCT FROM 'object'
    THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_run_invalid';
    END IF;

    INSERT INTO public.research_lab_weight_precompute_runs_v3 (
        precompute_run_id, storage_schema_version, protocol_schema_version,
        network_genesis_hash, netuid, epoch_id, epoch_ref,
        request_hash, planned_submission_block, release_commit_sha, release_manifest_hash, run_doc
    ) VALUES (
        p_precompute_run_id, 'leadpoet.weight_precompute_run.v3',
        'leadpoet.published_weight_bundle.v2', p_network_genesis_hash,
        p_netuid, p_epoch_id, p_epoch_ref, p_request_hash, p_planned_submission_block,
        p_release_commit_sha, p_release_manifest_hash, p_run_doc
    ) ON CONFLICT (precompute_run_id) DO NOTHING
    RETURNING * INTO inserted;
    IF FOUND THEN
        RETURN NEXT inserted;
        RETURN;
    END IF;

    SELECT * INTO existing
    FROM public.research_lab_weight_precompute_runs_v3
    WHERE precompute_run_id = p_precompute_run_id;
    IF NOT FOUND
       OR existing.network_genesis_hash IS DISTINCT FROM p_network_genesis_hash
       OR existing.netuid IS DISTINCT FROM p_netuid
       OR existing.epoch_id IS DISTINCT FROM p_epoch_id
       OR existing.epoch_ref IS DISTINCT FROM p_epoch_ref
       OR existing.request_hash IS DISTINCT FROM p_request_hash
       OR existing.planned_submission_block IS DISTINCT FROM p_planned_submission_block
       OR existing.release_commit_sha IS DISTINCT FROM p_release_commit_sha
       OR existing.release_manifest_hash IS DISTINCT FROM p_release_manifest_hash
       OR existing.run_doc IS DISTINCT FROM p_run_doc THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_run_replay_differs';
    END IF;
    RETURN NEXT existing;
END;
$$;

CREATE OR REPLACE FUNCTION public.record_research_lab_weight_precompute_input_set_v3(
    p_precompute_run_id UUID,
    p_input_set_hash TEXT,
    p_source_input_root TEXT,
    p_calculation_snapshot_hash TEXT,
    p_input_receipt_hashes JSONB,
    p_input_set_doc JSONB
)
RETURNS SETOF public.research_lab_weight_precompute_input_sets_v3
LANGUAGE plpgsql
VOLATILE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    run_row public.research_lab_weight_precompute_runs_v3%ROWTYPE;
    existing public.research_lab_weight_precompute_input_sets_v3%ROWTYPE;
    inserted public.research_lab_weight_precompute_input_sets_v3%ROWTYPE;
BEGIN
    IF p_precompute_run_id IS NULL
       OR p_input_set_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_source_input_root !~ '^sha256:[0-9a-f]{64}$'
       OR p_calculation_snapshot_hash !~ '^sha256:[0-9a-f]{64}$'
       OR NOT public.research_lab_weight_precompute_complete_input_set_v3(p_input_receipt_hashes)
       OR pg_catalog.jsonb_typeof(p_input_set_doc) IS DISTINCT FROM 'object'
    THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_input_set_invalid';
    END IF;
    SELECT * INTO run_row
    FROM public.research_lab_weight_precompute_runs_v3
    WHERE precompute_run_id = p_precompute_run_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_run_missing';
    END IF;

    INSERT INTO public.research_lab_weight_precompute_input_sets_v3 (
        precompute_run_id, storage_schema_version, protocol_schema_version,
        release_commit_sha, release_manifest_hash, input_set_hash,
        source_input_root, calculation_snapshot_hash, input_receipt_hashes, input_set_doc
    ) VALUES (
        p_precompute_run_id, 'leadpoet.weight_precompute_input_set.v3',
        run_row.protocol_schema_version, run_row.release_commit_sha,
        run_row.release_manifest_hash, p_input_set_hash,
        p_source_input_root, p_calculation_snapshot_hash, p_input_receipt_hashes, p_input_set_doc
    ) ON CONFLICT (precompute_run_id) DO NOTHING
    RETURNING * INTO inserted;
    IF FOUND THEN
        RETURN NEXT inserted;
        RETURN;
    END IF;

    SELECT * INTO existing
    FROM public.research_lab_weight_precompute_input_sets_v3
    WHERE precompute_run_id = p_precompute_run_id;
    IF NOT FOUND
       OR existing.input_set_hash IS DISTINCT FROM p_input_set_hash
       OR existing.source_input_root IS DISTINCT FROM p_source_input_root
       OR existing.calculation_snapshot_hash IS DISTINCT FROM p_calculation_snapshot_hash
       OR existing.input_receipt_hashes IS DISTINCT FROM p_input_receipt_hashes
       OR existing.input_set_doc IS DISTINCT FROM p_input_set_doc THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_input_set_replay_differs';
    END IF;
    RETURN NEXT existing;
END;
$$;

CREATE OR REPLACE FUNCTION public.append_research_lab_weight_precompute_stage_event_v3(
    p_stage_event_id UUID,
    p_precompute_run_id UUID,
    p_stage_sequence INTEGER,
    p_input_set_hash TEXT,
    p_stage_name TEXT,
    p_stage_status TEXT,
    p_event_hash TEXT,
    p_event_doc JSONB
)
RETURNS SETOF public.research_lab_weight_precompute_stage_events_v3
LANGUAGE plpgsql
VOLATILE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    existing public.research_lab_weight_precompute_stage_events_v3%ROWTYPE;
    inserted public.research_lab_weight_precompute_stage_events_v3%ROWTYPE;
BEGIN
    IF p_stage_event_id IS NULL
       OR p_precompute_run_id IS NULL
       OR p_stage_sequence < 0
       OR p_input_set_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_stage_name !~ '^[a-z][a-z0-9_.-]{0,127}$'
       OR p_stage_status NOT IN ('started', 'succeeded', 'failed')
       OR p_event_hash !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(p_event_doc) IS DISTINCT FROM 'object'
    THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_stage_event_invalid';
    END IF;
    INSERT INTO public.research_lab_weight_precompute_stage_events_v3 (
        stage_event_id, precompute_run_id, stage_sequence,
        storage_schema_version, input_set_hash, stage_name, stage_status,
        event_hash, event_doc
    ) VALUES (
        p_stage_event_id, p_precompute_run_id, p_stage_sequence,
        'leadpoet.weight_precompute_stage_event.v3', p_input_set_hash,
        p_stage_name, p_stage_status, p_event_hash, p_event_doc
    ) ON CONFLICT (stage_event_id) DO NOTHING
    RETURNING * INTO inserted;
    IF FOUND THEN
        RETURN NEXT inserted;
        RETURN;
    END IF;
    SELECT * INTO existing
    FROM public.research_lab_weight_precompute_stage_events_v3
    WHERE stage_event_id = p_stage_event_id;
    IF NOT FOUND
       OR existing.precompute_run_id IS DISTINCT FROM p_precompute_run_id
       OR existing.stage_sequence IS DISTINCT FROM p_stage_sequence
       OR existing.input_set_hash IS DISTINCT FROM p_input_set_hash
       OR existing.stage_name IS DISTINCT FROM p_stage_name
       OR existing.stage_status IS DISTINCT FROM p_stage_status
       OR existing.event_hash IS DISTINCT FROM p_event_hash
       OR existing.event_doc IS DISTINCT FROM p_event_doc THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_stage_event_replay_differs';
    END IF;
    RETURN NEXT existing;
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_weight_precompute_readback_v3(
    p_precompute_run_id UUID
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $$
DECLARE
    output_doc JSONB;
BEGIN
    SELECT pg_catalog.jsonb_build_object(
        'run', pg_catalog.to_jsonb(run_row),
        'complete_input_set', pg_catalog.to_jsonb(input_set_row),
        'stage_events', COALESCE(
            pg_catalog.jsonb_agg(pg_catalog.to_jsonb(stage_row)
                ORDER BY stage_row.stage_sequence, stage_row.occurred_at, stage_row.stage_event_id),
            '[]'::JSONB
        )
    ) INTO output_doc
    FROM public.research_lab_weight_precompute_runs_v3 AS run_row
    LEFT JOIN public.research_lab_weight_precompute_input_sets_v3 AS input_set_row
      ON input_set_row.precompute_run_id = run_row.precompute_run_id
    LEFT JOIN public.research_lab_weight_precompute_stage_events_v3 AS stage_row
      ON stage_row.precompute_run_id = run_row.precompute_run_id
    WHERE run_row.precompute_run_id = p_precompute_run_id
    GROUP BY run_row, input_set_row;
    IF output_doc IS NULL THEN
        RAISE EXCEPTION 'research_lab_weight_precompute_readback_missing';
    END IF;
    RETURN output_doc;
END;
$$;

CREATE OR REPLACE VIEW public.research_lab_weight_precompute_run_current_v3
WITH (security_invoker = true) AS
SELECT
    run_row.precompute_run_id,
    run_row.storage_schema_version,
    run_row.protocol_schema_version,
    run_row.network_genesis_hash,
    run_row.netuid,
    run_row.epoch_id,
    run_row.epoch_ref,
    run_row.request_hash,
    run_row.planned_submission_block,
    run_row.release_commit_sha,
    run_row.release_manifest_hash,
    input_set_row.input_set_hash,
    input_set_row.source_input_root,
    input_set_row.calculation_snapshot_hash,
    input_set_row.input_receipt_hashes,
    input_set_row.completed_at,
    run_row.created_at
FROM public.research_lab_weight_precompute_runs_v3 AS run_row
LEFT JOIN public.research_lab_weight_precompute_input_sets_v3 AS input_set_row
  ON input_set_row.precompute_run_id = run_row.precompute_run_id;

CREATE OR REPLACE FUNCTION public.research_lab_weight_precompute_store_contract_v3()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = pg_catalog
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'storage_schema_version', 'leadpoet.weight_precompute_store.v3',
        'protocol_schema_version', 'leadpoet.published_weight_bundle.v2',
        'append_only', true,
        'readback_rpc', 'research_lab_weight_precompute_readback_v3',
        'tables', pg_catalog.jsonb_build_array(
            'research_lab_weight_precompute_runs_v3',
            'research_lab_weight_precompute_input_sets_v3',
            'research_lab_weight_precompute_stage_events_v3'
        )
    )
$$;

REVOKE ALL ON TABLE
    public.research_lab_weight_precompute_runs_v3,
    public.research_lab_weight_precompute_input_sets_v3,
    public.research_lab_weight_precompute_stage_events_v3,
    public.research_lab_weight_precompute_run_current_v3
FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT ON TABLE
    public.research_lab_weight_precompute_runs_v3,
    public.research_lab_weight_precompute_input_sets_v3,
    public.research_lab_weight_precompute_stage_events_v3
TO service_role;
GRANT SELECT ON TABLE public.research_lab_weight_precompute_run_current_v3
TO service_role;

ALTER TABLE public.research_lab_weight_precompute_runs_v3 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_weight_precompute_input_sets_v3 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_weight_precompute_stage_events_v3 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_weight_precompute_runs_v3 FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_weight_precompute_input_sets_v3 FORCE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_weight_precompute_stage_events_v3 FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS research_lab_weight_precompute_runs_v3_service_select
    ON public.research_lab_weight_precompute_runs_v3;
CREATE POLICY research_lab_weight_precompute_runs_v3_service_select
    ON public.research_lab_weight_precompute_runs_v3
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS research_lab_weight_precompute_runs_v3_service_insert
    ON public.research_lab_weight_precompute_runs_v3;
CREATE POLICY research_lab_weight_precompute_runs_v3_service_insert
    ON public.research_lab_weight_precompute_runs_v3
    FOR INSERT TO service_role WITH CHECK (true);
DROP POLICY IF EXISTS research_lab_weight_precompute_input_sets_v3_service_select
    ON public.research_lab_weight_precompute_input_sets_v3;
CREATE POLICY research_lab_weight_precompute_input_sets_v3_service_select
    ON public.research_lab_weight_precompute_input_sets_v3
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS research_lab_weight_precompute_input_sets_v3_service_insert
    ON public.research_lab_weight_precompute_input_sets_v3;
CREATE POLICY research_lab_weight_precompute_input_sets_v3_service_insert
    ON public.research_lab_weight_precompute_input_sets_v3
    FOR INSERT TO service_role WITH CHECK (true);
DROP POLICY IF EXISTS research_lab_weight_precompute_stage_events_v3_service_select
    ON public.research_lab_weight_precompute_stage_events_v3;
CREATE POLICY research_lab_weight_precompute_stage_events_v3_service_select
    ON public.research_lab_weight_precompute_stage_events_v3
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS research_lab_weight_precompute_stage_events_v3_service_insert
    ON public.research_lab_weight_precompute_stage_events_v3;
CREATE POLICY research_lab_weight_precompute_stage_events_v3_service_insert
    ON public.research_lab_weight_precompute_stage_events_v3
    FOR INSERT TO service_role WITH CHECK (true);

REVOKE ALL ON FUNCTION public.research_lab_weight_precompute_sha256_map_v3(JSONB)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_weight_precompute_complete_input_set_v3(JSONB)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.prevent_research_lab_weight_precompute_mutation_v3()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.begin_research_lab_weight_precompute_run_v3(
    UUID, TEXT, INTEGER, BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.record_research_lab_weight_precompute_input_set_v3(
    UUID, TEXT, TEXT, TEXT, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.append_research_lab_weight_precompute_stage_event_v3(
    UUID, UUID, INTEGER, TEXT, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_weight_precompute_readback_v3(UUID)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_weight_precompute_store_contract_v3()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_weight_precompute_sha256_map_v3(JSONB)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_weight_precompute_complete_input_set_v3(JSONB)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.begin_research_lab_weight_precompute_run_v3(
    UUID, TEXT, INTEGER, BIGINT, TEXT, TEXT, BIGINT, TEXT, TEXT, JSONB
) TO service_role;
GRANT EXECUTE ON FUNCTION public.record_research_lab_weight_precompute_input_set_v3(
    UUID, TEXT, TEXT, TEXT, JSONB, JSONB
) TO service_role;
GRANT EXECUTE ON FUNCTION public.append_research_lab_weight_precompute_stage_event_v3(
    UUID, UUID, INTEGER, TEXT, TEXT, TEXT, TEXT, JSONB
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_weight_precompute_readback_v3(UUID)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_weight_precompute_store_contract_v3()
    TO service_role;

COMMENT ON TABLE public.research_lab_weight_precompute_runs_v3 IS
    'Append-only release-bound precompute runs for the external V2 weight protocol.';
COMMENT ON TABLE public.research_lab_weight_precompute_input_sets_v3 IS
    'One complete immutable gateway-owned V2 input-receipt frontier, source root, and snapshot hash per release-bound precompute run.';
COMMENT ON TABLE public.research_lab_weight_precompute_stage_events_v3 IS
    'Append-only, complete-input-set-bound precompute stage evidence.';
COMMENT ON FUNCTION public.research_lab_weight_precompute_readback_v3(UUID) IS
    'Returns exact stored run, complete input set, and ordered stage-event evidence.';

NOTIFY pgrst, 'reload schema';

COMMIT;
