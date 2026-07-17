-- Append-only proof that a signed pre-V2 allocation did not reach finalized
-- epoch-end chain state. These findings classify historical allocations
-- without creating payment credit or erasing the outstanding obligation.
-- Apply after scripts/102-research-lab-legacy-allocation-netuid-compat.sql.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_legacy_allocation_nonfinalizations_v2 (
    netuid                  INTEGER     NOT NULL CHECK (netuid > 0),
    epoch_id                INTEGER     NOT NULL CHECK (epoch_id >= 0),
    schema_version          TEXT        NOT NULL
                                       CHECK (
                                           schema_version =
                                           'leadpoet.legacy_allocation_nonfinalization.v2'
                                       ),
    allocation_hash         TEXT        NOT NULL UNIQUE
                                       CHECK (allocation_hash ~ '^sha256:[0-9a-f]{64}$'),
    finding_hash            TEXT        NOT NULL UNIQUE
                                       CHECK (finding_hash ~ '^sha256:[0-9a-f]{64}$'),
    finding_receipt_hash    TEXT        NOT NULL UNIQUE
                                       REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                       ON DELETE RESTRICT,
    allocation_doc          JSONB       NOT NULL CHECK (
                                       jsonb_typeof(allocation_doc) = 'object'
                                       AND allocation_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                       ),
    finding_doc             JSONB       NOT NULL CHECK (
                                       jsonb_typeof(finding_doc) = 'object'
                                       AND finding_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|raw_secret|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                       ),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (netuid, epoch_id),
    CHECK ((allocation_doc->>'epoch')::INTEGER = epoch_id),
    CHECK (
        CASE
            WHEN NOT (allocation_doc ? 'netuid') THEN TRUE
            WHEN allocation_doc->>'netuid' ~ '^[1-9][0-9]*$'
                THEN (allocation_doc->>'netuid')::NUMERIC = netuid
            ELSE FALSE
        END
    ),
    CHECK (allocation_doc->>'allocation_hash' = allocation_hash),
    CHECK (finding_doc->>'schema_version' = schema_version),
    CHECK ((finding_doc->>'epoch_id')::INTEGER = epoch_id),
    CHECK ((finding_doc->>'netuid')::INTEGER = netuid),
    CHECK (finding_doc->>'allocation_hash' = allocation_hash),
    CHECK (finding_doc->>'finding_hash' = finding_hash)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_legacy_nonfinalization_receipt_v2
    ON public.research_lab_legacy_allocation_nonfinalizations_v2(finding_receipt_hash);

DROP TRIGGER IF EXISTS prevent_research_lab_legacy_nonfinalization_v2_mutation
    ON public.research_lab_legacy_allocation_nonfinalizations_v2;
CREATE TRIGGER prevent_research_lab_legacy_nonfinalization_v2_mutation
    BEFORE UPDATE OR DELETE
    ON public.research_lab_legacy_allocation_nonfinalizations_v2
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

REVOKE ALL
    ON TABLE public.research_lab_legacy_allocation_nonfinalizations_v2
    FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_legacy_allocation_nonfinalizations_v2
    TO service_role;

ALTER TABLE public.research_lab_legacy_allocation_nonfinalizations_v2
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_legacy_allocation_nonfinalizations_v2;
CREATE POLICY service_role_read
    ON public.research_lab_legacy_allocation_nonfinalizations_v2
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_legacy_allocation_nonfinalizations_v2;
CREATE POLICY service_role_insert
    ON public.research_lab_legacy_allocation_nonfinalizations_v2
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_legacy_allocation_nonfinalizations_v2 IS
    'Append-only measured proof that a signed pre-V2 allocation differed from finalized epoch-end chain state and therefore creates no payment credit.';

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Verify after applying:
-- SELECT netuid, epoch_id, allocation_hash, finding_hash,
--        finding_receipt_hash
-- FROM public.research_lab_legacy_allocation_nonfinalizations_v2
-- ORDER BY epoch_id;
