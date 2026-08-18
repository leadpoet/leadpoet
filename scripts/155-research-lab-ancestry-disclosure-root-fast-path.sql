-- Bound ancestry disclosure lookup by the exact checkpoint root first.
--
-- Most certificate disclosures name the disclosed proof's own root. Resolve
-- that common case through the root primary key before retaining the original
-- lineage/sequence scan for valid non-root disclosures.

BEGIN;

SET LOCAL lock_timeout = '5s';

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
            -- Exact root disclosures use the checkpoint primary key. The
            -- fallback preserves the existing contract for a non-root receipt
            -- disclosed by an otherwise matching durable certificate.
            IF NOT EXISTS (
                SELECT 1
                FROM public.research_lab_attested_ancestry_checkpoints_v2 p,
                     LATERAL pg_catalog.jsonb_array_elements(
                         p.proof_doc->'disclosed_receipts'
                     ) disclosed
                WHERE p.root_receipt_hash = parent->>'parent_receipt_hash'
                  AND p.lineage_id = lineage
                  AND p.certificate_sequence = (parent->>'authority_sequence')::BIGINT
                  AND p.certificate_sequence < cert_sequence
                  AND disclosed->>'receipt_hash' = parent->>'parent_receipt_hash'
            ) THEN
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

CREATE OR REPLACE FUNCTION
public.research_lab_ancestry_disclosure_lookup_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
        'leadpoet.ancestry-disclosure-lookup-contract.v1',
        'persistence_rpc',
        'persist_research_lab_ancestry_checkpoint_v2',
        'root_witness_key',
        'root_receipt_hash',
        'non_root_fallback',
        'lineage_sequence_disclosure_scan'
    );
$$;

REVOKE ALL ON FUNCTION
    public.persist_research_lab_ancestry_checkpoint_v2(JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.persist_research_lab_ancestry_checkpoint_v2(JSONB)
    TO service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_ancestry_disclosure_lookup_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_ancestry_disclosure_lookup_contract_v1()
    TO service_role;

COMMENT ON FUNCTION
public.research_lab_ancestry_disclosure_lookup_contract_v1() IS
    'Read-only contract proving exact-root ancestry disclosure lookup precedes the legacy non-root fallback.';

NOTIFY pgrst, 'reload schema';

COMMIT;
