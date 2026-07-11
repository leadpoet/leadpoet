-- Separate SOURCE_ADD allocation accounting and enforce future public score-bundle containment.
--
-- Existing allocation rows remain immutable. Historical score bundles that
-- contain private image pointers remain readable, while the NOT VALID check
-- rejects that shape on every new insert.
-- Legacy SOURCE_ADD rows embedded in champion_allocations are retained for
-- audit only; runtime replay counts first-class source_add_allocations rows.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_emission_allocation_snapshots
    ADD COLUMN IF NOT EXISTS source_add_alpha_percent NUMERIC(10, 6) NOT NULL DEFAULT 0
    CHECK (source_add_alpha_percent >= 0 AND source_add_alpha_percent <= 100);

ALTER TABLE public.research_lab_emission_allocation_snapshots
    DROP CONSTRAINT IF EXISTS research_lab_emission_allocation_source_add_cap_check;

ALTER TABLE public.research_lab_emission_allocation_snapshots
    ADD CONSTRAINT research_lab_emission_allocation_source_add_cap_check
    CHECK (
        source_add_alpha_percent
        + reimbursement_alpha_percent
        + champion_alpha_percent
        + queued_champion_alpha_percent
        + unallocated_alpha_percent
        <= lab_cap_alpha_percent + 0.000001
    ) NOT VALID;

ALTER TABLE public.research_lab_emission_allocation_snapshots
    VALIDATE CONSTRAINT research_lab_emission_allocation_source_add_cap_check;

CREATE OR REPLACE VIEW public.research_lab_emission_allocation_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (epoch, netuid, policy_id)
    allocation_id,
    schema_version,
    epoch,
    netuid,
    policy_id,
    snapshot_status,
    lab_cap_alpha_percent,
    reimbursement_alpha_percent,
    champion_alpha_percent,
    queued_champion_alpha_percent,
    unallocated_alpha_percent,
    input_hash,
    allocation_hash,
    allocation_doc,
    created_at,
    source_add_alpha_percent
FROM public.research_lab_emission_allocation_snapshots
WHERE snapshot_status IN ('shadow', 'candidate', 'active')
ORDER BY epoch, netuid, policy_id, created_at DESC;

REVOKE ALL ON TABLE public.research_lab_emission_allocation_current FROM anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_emission_allocation_current TO service_role;

ALTER TABLE public.research_evaluation_score_bundles
    DROP CONSTRAINT IF EXISTS research_evaluation_score_bundles_public_containment_check;

ALTER TABLE public.research_evaluation_score_bundles
    ADD CONSTRAINT research_evaluation_score_bundles_public_containment_check
    CHECK (
        score_bundle_doc::TEXT !~* '(image[_-]?digest|manifest[_-]?uri|image[_-]?repository|\.dkr\.ecr\.|private_model_manifest_doc|candidate_patch_manifest|proxy[_-]?url|://[^/]+:[^/@]+@)'
    ) NOT VALID;

COMMENT ON COLUMN public.research_lab_emission_allocation_snapshots.source_add_alpha_percent IS
    'First-priority SOURCE_ADD percentage deducted from the configured Research Lab cap before reimbursement/champion allocation.';

COMMENT ON CONSTRAINT research_evaluation_score_bundles_public_containment_check
    ON public.research_evaluation_score_bundles IS
    'New public score bundles must contain hashes/commit identity only, never raw ECR image digests, private manifests, patch manifests, or credentialed URLs.';

COMMIT;
