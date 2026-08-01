-- Authenticated, bounded V2 receipt ancestry and compact weight authority.
--
-- Raw receipts, attempts, host operations, and edges remain append-only.  A
-- checkpoint stores only one enclave-signed local delta commitment and its
-- bounded disclosure. Once a receipt root has a durable certificate, that
-- root can never be reintroduced as a full-graph parent. Independent legacy
-- branches can perform their one finite bootstrap without blocking each other.

BEGIN;

CREATE TABLE IF NOT EXISTS
public.research_lab_attested_ancestry_checkpoints_v2 (
    root_receipt_hash         TEXT PRIMARY KEY
        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
        ON DELETE RESTRICT,
    schema_version            TEXT NOT NULL
        CHECK (schema_version = 'leadpoet.attested_ancestry_certificate.v2'),
    lineage_id                TEXT NOT NULL
        CHECK (lineage_id ~ '^sha256:[0-9a-f]{64}$'),
    certificate_hash          TEXT NOT NULL UNIQUE
        CHECK (certificate_hash ~ '^sha256:[0-9a-f]{64}$'),
    certificate_sequence      BIGINT NOT NULL CHECK (certificate_sequence >= 0),
    issuer_boot_identity_hash TEXT NOT NULL
        REFERENCES public.research_lab_attested_boot_identities_v2(boot_identity_hash)
        ON DELETE RESTRICT,
    proof_hash                TEXT NOT NULL UNIQUE
        CHECK (proof_hash ~ '^sha256:[0-9a-f]{64}$'),
    checkpoint_graph_hash     TEXT NOT NULL UNIQUE
        CHECK (checkpoint_graph_hash ~ '^sha256:[0-9a-f]{64}$'),
    certificate_doc           JSONB NOT NULL CHECK (
        jsonb_typeof(certificate_doc) = 'object'
        AND certificate_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
    ),
    proof_doc                 JSONB NOT NULL CHECK (
        jsonb_typeof(proof_doc) = 'object'
        AND proof_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
    ),
    checkpoint_graph_doc      JSONB NOT NULL CHECK (
        jsonb_typeof(checkpoint_graph_doc) = 'object'
        AND checkpoint_graph_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
    ),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (certificate_doc->>'schema_version' = schema_version),
    CHECK (certificate_doc->>'certificate_hash' = certificate_hash),
    CHECK (certificate_doc #>> '{claim,output_root_receipt_hash}' = root_receipt_hash),
    CHECK (certificate_doc #>> '{claim,lineage_id}' = lineage_id),
    CHECK ((certificate_doc #>> '{claim,certificate_sequence}')::BIGINT = certificate_sequence),
    CHECK (certificate_doc #>> '{claim,issuer_boot_identity_hash}' = issuer_boot_identity_hash),
    CHECK (proof_doc->>'schema_version' = 'leadpoet.attested_ancestry_compact_proof.v2'),
    CHECK (proof_doc->>'proof_hash' = proof_hash),
    CHECK (proof_doc #>> '{certificate,schema_version}' = schema_version),
    CHECK (proof_doc #>> '{certificate,certificate_hash}' = certificate_hash),
    CHECK (proof_doc #>> '{certificate,claim,output_root_receipt_hash}' = root_receipt_hash),
    CHECK (proof_doc #>> '{certificate,claim,lineage_id}' = lineage_id),
    CHECK ((proof_doc #>> '{certificate,claim,certificate_sequence}')::BIGINT = certificate_sequence),
    CHECK (proof_doc #>> '{certificate,claim,issuer_boot_identity_hash}' = issuer_boot_identity_hash),
    CHECK (checkpoint_graph_doc->>'schema_version' = 'leadpoet.attested_checkpointed_receipt_graph.v3'),
    CHECK (checkpoint_graph_doc->>'root_receipt_hash' = root_receipt_hash),
    CHECK (checkpoint_graph_doc->>'ancestry_lineage_id' = lineage_id),
    CHECK (checkpoint_graph_doc #>> '{ancestry_proof,proof_hash}' = proof_hash),
    CHECK (checkpoint_graph_doc #>> '{ancestry_proof,certificate,certificate_hash}' = certificate_hash),
    CHECK (checkpoint_graph_doc #>> '{ancestry_proof,certificate,claim,output_root_receipt_hash}' = root_receipt_hash),
    CHECK (checkpoint_graph_doc #>> '{ancestry_proof,certificate,claim,lineage_id}' = lineage_id),
    UNIQUE (root_receipt_hash, lineage_id)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_ancestry_checkpoint_lineage_v2
    ON public.research_lab_attested_ancestry_checkpoints_v2(
        lineage_id, certificate_sequence DESC
    );

CREATE INDEX IF NOT EXISTS idx_research_lab_ancestry_checkpoint_issuer_v2
    ON public.research_lab_attested_ancestry_checkpoints_v2(
        issuer_boot_identity_hash, certificate_sequence DESC
    );

CREATE TABLE IF NOT EXISTS
public.research_lab_attested_ancestry_activations_v2 (
    lineage_id               TEXT NOT NULL
        CHECK (lineage_id ~ '^sha256:[0-9a-f]{64}$'),
    activation_root_receipt_hash TEXT PRIMARY KEY
        REFERENCES public.research_lab_attested_ancestry_checkpoints_v2(root_receipt_hash)
        ON DELETE RESTRICT,
    activation_certificate_hash TEXT NOT NULL UNIQUE
        REFERENCES public.research_lab_attested_ancestry_checkpoints_v2(certificate_hash)
        ON DELETE RESTRICT,
    activated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (lineage_id, activation_root_receipt_hash)
);

CREATE OR REPLACE FUNCTION
public.persist_research_lab_ancestry_checkpoint_v2(checkpoint JSONB)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    lineage TEXT;
    root_hash TEXT;
    cert_hash TEXT;
    cert_sequence BIGINT;
    issuer_hash TEXT;
    proof_hash_value TEXT;
    graph_hash_value TEXT;
    parent JSONB;
    parent_kind TEXT;
    stored public.research_lab_attested_ancestry_checkpoints_v2%ROWTYPE;
BEGIN
    IF jsonb_typeof(checkpoint) <> 'object' THEN
        RAISE EXCEPTION 'checkpoint must be an object' USING ERRCODE = '22023';
    END IF;
    lineage := checkpoint->>'lineage_id';
    root_hash := checkpoint->>'root_receipt_hash';
    cert_hash := checkpoint->>'certificate_hash';
    cert_sequence := (checkpoint->>'certificate_sequence')::BIGINT;
    issuer_hash := checkpoint->>'issuer_boot_identity_hash';
    proof_hash_value := checkpoint->>'proof_hash';
    graph_hash_value := checkpoint->>'checkpoint_graph_hash';
    IF lineage !~ '^sha256:[0-9a-f]{64}$'
       OR root_hash !~ '^sha256:[0-9a-f]{64}$'
       OR cert_hash !~ '^sha256:[0-9a-f]{64}$'
       OR proof_hash_value !~ '^sha256:[0-9a-f]{64}$'
       OR graph_hash_value !~ '^sha256:[0-9a-f]{64}$'
       OR issuer_hash !~ '^sha256:[0-9a-f]{64}$'
       OR cert_sequence < 0
       OR jsonb_typeof(checkpoint->'certificate_doc') <> 'object'
       OR jsonb_typeof(checkpoint->'proof_doc') <> 'object'
       OR jsonb_typeof(checkpoint->'checkpoint_graph_doc') <> 'object'
       OR jsonb_typeof(checkpoint #> '{certificate_doc,claim,parent_authorities}') <> 'array'
    THEN
        RAISE EXCEPTION 'checkpoint contract is invalid' USING ERRCODE = '22023';
    END IF;

    -- Serialize the activation decision for one lineage without imposing a
    -- mutable global head; independent sibling certificates remain valid.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(lineage, 0)
    );
    FOR parent IN
        SELECT value
        FROM pg_catalog.jsonb_array_elements(
            checkpoint #> '{certificate_doc,claim,parent_authorities}'
        )
    LOOP
        parent_kind := parent->>'authority_kind';
        IF parent_kind = 'full_projection' THEN
            IF EXISTS (
                SELECT 1
                FROM public.research_lab_attested_ancestry_activations_v2 a
                WHERE a.lineage_id = lineage
                  AND a.activation_root_receipt_hash = parent->>'parent_receipt_hash'
            ) THEN
                RAISE EXCEPTION 'compacted ancestry root rejects full graph parent'
                    USING ERRCODE = '23514';
            END IF;
        ELSIF parent_kind = 'certificate' THEN
            IF NOT EXISTS (
                SELECT 1
                FROM public.research_lab_attested_ancestry_checkpoints_v2 p
                WHERE p.root_receipt_hash = parent->>'parent_receipt_hash'
                  AND p.lineage_id = lineage
                  AND p.certificate_hash = parent->>'authority_hash'
                  AND p.certificate_sequence = (parent->>'authority_sequence')::BIGINT
                  AND p.certificate_sequence < cert_sequence
            ) THEN
                RAISE EXCEPTION 'checkpoint certificate parent is not durable'
                    USING ERRCODE = '23503';
            END IF;
        ELSIF parent_kind = 'certificate_disclosure' THEN
            IF NOT EXISTS (
                SELECT 1
                FROM public.research_lab_attested_ancestry_checkpoints_v2 p,
                     LATERAL pg_catalog.jsonb_array_elements(
                         p.proof_doc->'disclosed_receipts'
                     ) disclosed
                WHERE p.lineage_id = lineage
                  AND p.certificate_sequence = (parent->>'authority_sequence')::BIGINT
                  AND p.certificate_sequence < cert_sequence
                  AND disclosed->>'receipt_hash' = parent->>'parent_receipt_hash'
            ) THEN
                RAISE EXCEPTION 'checkpoint disclosure parent is not durable'
                    USING ERRCODE = '23503';
            END IF;
        ELSE
            RAISE EXCEPTION 'checkpoint parent authority kind is invalid'
                USING ERRCODE = '22023';
        END IF;
    END LOOP;

    INSERT INTO public.research_lab_attested_ancestry_checkpoints_v2 (
        root_receipt_hash, schema_version, lineage_id, certificate_hash,
        certificate_sequence, issuer_boot_identity_hash, proof_hash,
        checkpoint_graph_hash, certificate_doc, proof_doc,
        checkpoint_graph_doc
    ) VALUES (
        root_hash, checkpoint->>'schema_version', lineage, cert_hash,
        cert_sequence, issuer_hash, proof_hash_value,
        graph_hash_value, checkpoint->'certificate_doc',
        checkpoint->'proof_doc', checkpoint->'checkpoint_graph_doc'
    ) ON CONFLICT (root_receipt_hash) DO NOTHING;

    SELECT * INTO stored
    FROM public.research_lab_attested_ancestry_checkpoints_v2 c
    WHERE c.root_receipt_hash = root_hash;
    IF stored.root_receipt_hash IS NULL
       OR stored.schema_version <> checkpoint->>'schema_version'
       OR stored.lineage_id <> lineage
       OR stored.certificate_hash <> cert_hash
       OR stored.certificate_sequence <> cert_sequence
       OR stored.issuer_boot_identity_hash <> issuer_hash
       OR stored.proof_hash <> proof_hash_value
       OR stored.checkpoint_graph_hash <> graph_hash_value
       OR stored.certificate_doc <> checkpoint->'certificate_doc'
       OR stored.proof_doc <> checkpoint->'proof_doc'
       OR stored.checkpoint_graph_doc <> checkpoint->'checkpoint_graph_doc'
    THEN
        RAISE EXCEPTION 'checkpoint durable readback conflicts'
            USING ERRCODE = '23505';
    END IF;

    -- Every newly certified root is an irreversible compaction frontier. A
    -- different legacy root may still perform its own one-time bootstrap.
    INSERT INTO public.research_lab_attested_ancestry_activations_v2 (
        lineage_id, activation_root_receipt_hash,
        activation_certificate_hash
    ) VALUES (lineage, root_hash, cert_hash)
    ON CONFLICT (activation_root_receipt_hash) DO NOTHING;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_ancestry_activations_v2 a
        WHERE a.lineage_id = lineage
          AND a.activation_root_receipt_hash = root_hash
          AND a.activation_certificate_hash = cert_hash
    ) THEN
        RAISE EXCEPTION 'ancestry root activation conflicts'
            USING ERRCODE = '23505';
    END IF;

    RETURN pg_catalog.jsonb_build_object(
        'status', 'persisted',
        'root_receipt_hash', stored.root_receipt_hash,
        'lineage_id', stored.lineage_id,
        'certificate_hash', stored.certificate_hash,
        'certificate_sequence', stored.certificate_sequence,
        'proof_hash', stored.proof_hash,
        'checkpoint_graph_hash', stored.checkpoint_graph_hash,
        'root_activated', EXISTS (
            SELECT 1
            FROM public.research_lab_attested_ancestry_activations_v2 a
            WHERE a.lineage_id = lineage
              AND a.activation_root_receipt_hash = root_hash
              AND a.activation_certificate_hash = cert_hash
        )
    );
END;
$$;

CREATE TABLE IF NOT EXISTS
public.research_lab_compact_weight_submissions_v2 (
    compact_submission_hash TEXT PRIMARY KEY
        CHECK (compact_submission_hash ~ '^sha256:[0-9a-f]{64}$'),
    bundle_hash             TEXT NOT NULL UNIQUE
        CHECK (bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    netuid                  INTEGER NOT NULL CHECK (netuid > 0),
    epoch_id                BIGINT NOT NULL CHECK (epoch_id >= 0),
    validator_hotkey        TEXT NOT NULL CHECK (length(validator_hotkey) BETWEEN 1 AND 128),
    lineage_id              TEXT NOT NULL CHECK (lineage_id ~ '^sha256:[0-9a-f]{64}$'),
    binding_receipt_hash    TEXT NOT NULL
        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
        ON DELETE RESTRICT,
    submission_doc          JSONB NOT NULL CHECK (
        jsonb_typeof(submission_doc) = 'object'
        AND submission_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|proxy-authorization|"authorization"[[:space:]]*:[[:space:]]*"[[:space:]]*(basic|bearer)[[:space:]]|://[^/]+:[^/@]+@)'
    ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (netuid, epoch_id, validator_hotkey),
    UNIQUE (bundle_hash, compact_submission_hash),
    FOREIGN KEY (binding_receipt_hash, lineage_id)
        REFERENCES public.research_lab_attested_ancestry_checkpoints_v2(
            root_receipt_hash, lineage_id
        ) ON DELETE RESTRICT,
    CHECK (submission_doc->>'schema_version' = 'leadpoet.compact_weight_submission.v2'),
    CHECK (submission_doc->>'compact_submission_hash' = compact_submission_hash),
    CHECK ((submission_doc #>> '{weight_result,netuid}')::INTEGER = netuid),
    CHECK ((submission_doc #>> '{weight_result,epoch_id}')::BIGINT = epoch_id),
    CHECK (submission_doc->>'validator_hotkey' = validator_hotkey),
    CHECK (submission_doc #>> '{binding_receipt,receipt_hash}' = binding_receipt_hash),
    CHECK (submission_doc #>> '{validator_ancestry_proof,certificate,claim,output_root_receipt_hash}' = binding_receipt_hash),
    CHECK (submission_doc #>> '{validator_ancestry_proof,certificate,claim,lineage_id}' = lineage_id)
);

CREATE TABLE IF NOT EXISTS
public.research_lab_compact_weight_publication_intents_v2 (
    bundle_hash             TEXT PRIMARY KEY
        CHECK (bundle_hash ~ '^sha256:[0-9a-f]{64}$'),
    compact_submission_hash TEXT NOT NULL,
    netuid                  INTEGER NOT NULL CHECK (netuid > 0),
    epoch_id                BIGINT NOT NULL CHECK (epoch_id >= 0),
    validator_hotkey        TEXT NOT NULL
        CHECK (length(validator_hotkey) BETWEEN 1 AND 128),
    root_receipt_hash       TEXT NOT NULL
        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
        ON DELETE RESTRICT,
    durable_readback_hash   TEXT NOT NULL
        CHECK (durable_readback_hash ~ '^sha256:[0-9a-f]{64}$'),
    transparency_event_hash TEXT NOT NULL
        CHECK (transparency_event_hash ~ '^sha256:[0-9a-f]{64}$'),
    epoch_authority_hash    TEXT NOT NULL
        CHECK (epoch_authority_hash ~ '^sha256:[0-9a-f]{64}$'),
    intent_hash             TEXT NOT NULL UNIQUE
        CHECK (intent_hash ~ '^sha256:[0-9a-f]{64}$'),
    intent_doc              JSONB NOT NULL CHECK (
        jsonb_typeof(intent_doc) = 'object'
        AND intent_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|proxy-authorization|"authorization"[[:space:]]*:[[:space:]]*"[[:space:]]*(basic|bearer)[[:space:]]|://[^/]+:[^/@]+@)'
    ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (bundle_hash, compact_submission_hash)
        REFERENCES public.research_lab_compact_weight_submissions_v2(
            bundle_hash, compact_submission_hash
        ) ON DELETE RESTRICT,
    CHECK (intent_doc->>'schema_version' = 'leadpoet.compact_weight_publication_intent.v2'),
    CHECK (intent_doc->>'bundle_hash' = bundle_hash),
    CHECK (intent_doc->>'compact_submission_hash' = compact_submission_hash),
    CHECK ((intent_doc->>'netuid')::INTEGER = netuid),
    CHECK ((intent_doc->>'epoch_id')::BIGINT = epoch_id),
    CHECK (intent_doc->>'validator_hotkey' = validator_hotkey),
    CHECK (intent_doc->>'root_receipt_hash' = root_receipt_hash),
    CHECK (intent_doc->>'durable_readback_hash' = durable_readback_hash),
    CHECK (intent_doc->>'transparency_event_hash' = transparency_event_hash),
    CHECK (intent_doc->>'epoch_authority_hash' = epoch_authority_hash),
    CHECK (intent_doc->>'intent_hash' = intent_hash)
);

CREATE TABLE IF NOT EXISTS
public.research_lab_compact_weight_authorities_v2 (
    bundle_hash              TEXT NOT NULL,
    compact_submission_hash  TEXT NOT NULL,
    netuid                   INTEGER NOT NULL CHECK (netuid > 0),
    epoch_id                 BIGINT NOT NULL CHECK (epoch_id >= 0),
    validator_hotkey         TEXT NOT NULL CHECK (length(validator_hotkey) BETWEEN 1 AND 128),
    authority_stage          TEXT NOT NULL CHECK (authority_stage IN ('published', 'finalized')),
    schema_version           TEXT NOT NULL CHECK (schema_version = 'leadpoet.compact_published_weight_authority.v2'),
    lineage_id               TEXT NOT NULL CHECK (lineage_id ~ '^sha256:[0-9a-f]{64}$'),
    authority_hash           TEXT NOT NULL UNIQUE CHECK (authority_hash ~ '^sha256:[0-9a-f]{64}$'),
    publication_receipt_hash TEXT NOT NULL
        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
        ON DELETE RESTRICT,
    compact_finalization_hash TEXT CHECK (
        compact_finalization_hash IS NULL
        OR compact_finalization_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    finalization_receipt_hash TEXT
        REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
        ON DELETE RESTRICT,
    authority_doc            JSONB NOT NULL CHECK (
        jsonb_typeof(authority_doc) = 'object'
        AND authority_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|proxy-authorization|"authorization"[[:space:]]*:[[:space:]]*"[[:space:]]*(basic|bearer)[[:space:]]|://[^/]+:[^/@]+@)'
    ),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (bundle_hash, authority_stage),
    UNIQUE (netuid, epoch_id, validator_hotkey, authority_stage),
    FOREIGN KEY (bundle_hash, compact_submission_hash)
        REFERENCES public.research_lab_compact_weight_submissions_v2(
            bundle_hash, compact_submission_hash
        ) ON DELETE RESTRICT,
    FOREIGN KEY (publication_receipt_hash, lineage_id)
        REFERENCES public.research_lab_attested_ancestry_checkpoints_v2(
            root_receipt_hash, lineage_id
        ) ON DELETE RESTRICT,
    FOREIGN KEY (finalization_receipt_hash, lineage_id)
        REFERENCES public.research_lab_attested_ancestry_checkpoints_v2(
            root_receipt_hash, lineage_id
        ) ON DELETE RESTRICT,
    CHECK (authority_doc->>'schema_version' = schema_version),
    CHECK (authority_doc->>'authority_stage' = authority_stage),
    CHECK (authority_doc->>'lineage_id' = lineage_id),
    CHECK (authority_doc->>'bundle_hash' = bundle_hash),
    CHECK (authority_doc->>'authority_hash' = authority_hash),
    CHECK (authority_doc #>> '{compact_submission,compact_submission_hash}' = compact_submission_hash),
    CHECK (authority_doc #>> '{compact_submission,validator_ancestry_proof,certificate,claim,lineage_id}' = lineage_id),
    CHECK (authority_doc #>> '{publication,publication_receipt_hash}' = publication_receipt_hash),
    CHECK (authority_doc #>> '{publication,ancestry_proof,certificate,claim,output_root_receipt_hash}' = publication_receipt_hash),
    CHECK (authority_doc #>> '{publication,ancestry_proof,certificate,claim,lineage_id}' = lineage_id),
    CHECK (
        (authority_stage = 'published'
         AND authority_doc->'finalization' = 'null'::JSONB
         AND compact_finalization_hash IS NULL
         AND finalization_receipt_hash IS NULL)
        OR
        (authority_stage = 'finalized'
         AND jsonb_typeof(authority_doc->'finalization') = 'object'
         AND authority_doc #>> '{finalization,compact_submission,compact_finalization_hash}' = compact_finalization_hash
         AND authority_doc #>> '{finalization,compact_submission,validator_receipt_delta,root_receipt_hash}' = finalization_receipt_hash
         AND authority_doc #>> '{finalization,compact_submission,validator_ancestry_proof,certificate,claim,output_root_receipt_hash}' = finalization_receipt_hash
         AND authority_doc #>> '{finalization,compact_submission,validator_ancestry_proof,certificate,claim,lineage_id}' = lineage_id)
    )
);

-- Preserve idempotency if an operator rehearsed an earlier candidate of this
-- not-yet-activated migration. The added foreign key and stage binding remain
-- fail-closed and never rewrite an existing authority row.
ALTER TABLE public.research_lab_compact_weight_authorities_v2
    ADD COLUMN IF NOT EXISTS finalization_receipt_hash TEXT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint
        WHERE conname = 'research_lab_compact_weight_authority_final_receipt_fk_v2'
          AND conrelid = 'public.research_lab_compact_weight_authorities_v2'::REGCLASS
    ) THEN
        ALTER TABLE public.research_lab_compact_weight_authorities_v2
            ADD CONSTRAINT research_lab_compact_weight_authority_final_receipt_fk_v2
            FOREIGN KEY (finalization_receipt_hash)
            REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
            ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_constraint
        WHERE conname = 'research_lab_compact_weight_authority_final_receipt_ch_v2'
          AND conrelid = 'public.research_lab_compact_weight_authorities_v2'::REGCLASS
    ) THEN
        ALTER TABLE public.research_lab_compact_weight_authorities_v2
            ADD CONSTRAINT research_lab_compact_weight_authority_final_receipt_ch_v2
            CHECK (
                (authority_stage = 'published'
                 AND finalization_receipt_hash IS NULL)
                OR
                (authority_stage = 'finalized'
                 AND finalization_receipt_hash IS NOT NULL
                 AND authority_doc #>> '{finalization,compact_submission,validator_receipt_delta,root_receipt_hash}' = finalization_receipt_hash)
            );
    END IF;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_research_lab_compact_weight_identity_v2
    ON public.research_lab_compact_weight_authorities_v2(
        netuid, epoch_id, validator_hotkey, authority_stage
    );

DO $$
DECLARE table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'research_lab_attested_ancestry_checkpoints_v2',
        'research_lab_attested_ancestry_activations_v2',
        'research_lab_compact_weight_submissions_v2',
        'research_lab_compact_weight_publication_intents_v2',
        'research_lab_compact_weight_authorities_v2'
    ] LOOP
        EXECUTE pg_catalog.format(
            'DROP TRIGGER IF EXISTS prevent_research_lab_bounded_v2_mutation ON public.%I',
            table_name
        );
        EXECUTE pg_catalog.format(
            'CREATE TRIGGER prevent_research_lab_bounded_v2_mutation BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation()',
            table_name
        );
        EXECUTE pg_catalog.format(
            'REVOKE ALL ON TABLE public.%I FROM PUBLIC, anon, authenticated',
            table_name
        );
        IF table_name IN (
            'research_lab_attested_ancestry_checkpoints_v2',
            'research_lab_attested_ancestry_activations_v2'
        ) THEN
            EXECUTE pg_catalog.format(
                'GRANT SELECT ON TABLE public.%I TO service_role', table_name
            );
        ELSE
            EXECUTE pg_catalog.format(
                'GRANT SELECT, INSERT ON TABLE public.%I TO service_role',
                table_name
            );
        END IF;
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name
        );
        EXECUTE pg_catalog.format(
            'DROP POLICY IF EXISTS service_role_read ON public.%I', table_name
        );
        EXECUTE pg_catalog.format(
            'CREATE POLICY service_role_read ON public.%I FOR SELECT TO service_role USING (true)',
            table_name
        );
        EXECUTE pg_catalog.format(
            'DROP POLICY IF EXISTS service_role_insert ON public.%I', table_name
        );
        EXECUTE pg_catalog.format(
            'CREATE POLICY service_role_insert ON public.%I FOR INSERT TO service_role WITH CHECK (true)',
            table_name
        );
    END LOOP;
END;
$$;

REVOKE ALL ON FUNCTION
    public.persist_research_lab_ancestry_checkpoint_v2(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.persist_research_lab_ancestry_checkpoint_v2(JSONB)
    TO service_role;

COMMENT ON TABLE public.research_lab_attested_ancestry_checkpoints_v2 IS
    'Append-only enclave-signed local ancestry deltas and bounded proofs.';
COMMENT ON TABLE public.research_lab_attested_ancestry_activations_v2 IS
    'Irreversible per-root frontier after which that root cannot expand to a full graph.';
COMMENT ON TABLE public.research_lab_compact_weight_submissions_v2 IS
    'First-class compact canonical weight submissions; no legacy full-bundle dependency.';
COMMENT ON TABLE public.research_lab_compact_weight_publication_intents_v2 IS
    'Append-only validated epoch and transparency boundary for retryable compact publication.';

NOTIFY pgrst, 'reload schema';

COMMIT;
