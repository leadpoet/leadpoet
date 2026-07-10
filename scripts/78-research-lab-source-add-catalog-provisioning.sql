-- SOURCE_ADD catalog/provisioning state and private duplicate identity.

ALTER TABLE public.research_lab_source_add_submissions
    ADD COLUMN IF NOT EXISTS source_identity_hash TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_research_lab_source_add_submissions_identity
    ON public.research_lab_source_add_submissions (source_identity_hash)
    WHERE source_identity_hash <> '';

ALTER TABLE public.research_lab_source_catalog
    ADD COLUMN IF NOT EXISTS source_identity_hash TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_research_lab_source_catalog_identity
    ON public.research_lab_source_catalog (source_identity_hash)
    WHERE source_identity_hash <> '';

CREATE OR REPLACE VIEW public.research_lab_source_add_submission_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (submission_id)
    *
FROM public.research_lab_source_add_submissions
ORDER BY submission_id, seq DESC, created_at DESC;

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_provisioning_events (
    provision_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provision_ref TEXT NOT NULL CHECK (provision_ref ~ '^source_add_provision:[0-9a-f]{16}$'),
    catalog_id TEXT NOT NULL REFERENCES public.research_lab_source_catalog(catalog_id) ON DELETE RESTRICT,
    submission_id TEXT NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    source_identity_hash TEXT NOT NULL DEFAULT '',
    registry_provider_id TEXT NOT NULL,
    provision_status TEXT NOT NULL CHECK (
        provision_status IN ('approved_pending_provision', 'provisioned_autoresearch_eligible', 'disabled')
    ),
    seq INTEGER NOT NULL CHECK (seq >= 0),
    provision_doc JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(provision_doc) = 'object'
        AND provision_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|"password"|raw_credential)'
    ),
    credential_envelope JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(credential_envelope) = 'object'
        AND credential_envelope::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|"password"|raw_credential)'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (adapter_id, seq),
    UNIQUE (provision_ref)
);

CREATE INDEX IF NOT EXISTS idx_research_lab_source_add_provisioning_current_lookup
    ON public.research_lab_source_add_provisioning_events (provision_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_lab_source_add_provisioning_identity
    ON public.research_lab_source_add_provisioning_events (source_identity_hash)
    WHERE source_identity_hash <> '';

REVOKE ALL ON TABLE public.research_lab_source_add_provisioning_events FROM anon;
REVOKE ALL ON TABLE public.research_lab_source_add_provisioning_events FROM authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_source_add_provisioning_events TO service_role;

ALTER TABLE public.research_lab_source_add_provisioning_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS research_lab_source_add_provisioning_service_select
    ON public.research_lab_source_add_provisioning_events;
CREATE POLICY research_lab_source_add_provisioning_service_select
    ON public.research_lab_source_add_provisioning_events
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS research_lab_source_add_provisioning_service_insert
    ON public.research_lab_source_add_provisioning_events;
CREATE POLICY research_lab_source_add_provisioning_service_insert
    ON public.research_lab_source_add_provisioning_events
    FOR INSERT TO service_role WITH CHECK (true);

CREATE OR REPLACE VIEW public.research_lab_source_add_provisioning_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (p.adapter_id)
    p.provision_event_id,
    p.provision_ref,
    p.catalog_id,
    p.submission_id,
    p.adapter_id,
    p.miner_hotkey,
    p.source_identity_hash,
    p.registry_provider_id,
    p.provision_status,
    p.seq,
    p.provision_doc,
    p.credential_envelope,
    p.created_at,
    c.source_name,
    c.source_kind,
    c.declared_base_domains,
    c.accepted_at,
    c.catalog_doc
FROM public.research_lab_source_add_provisioning_events p
JOIN public.research_lab_source_catalog c ON c.catalog_id = p.catalog_id
ORDER BY p.adapter_id, p.seq DESC, p.created_at DESC;

REVOKE ALL ON TABLE public.research_lab_source_add_provisioning_current FROM anon;
REVOKE ALL ON TABLE public.research_lab_source_add_provisioning_current FROM authenticated;
GRANT SELECT ON TABLE public.research_lab_source_add_provisioning_current TO service_role;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_source_add_provisioning_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'research_lab_source_add_provisioning_events is append-only';
END;
$$;

DROP TRIGGER IF EXISTS trg_research_lab_source_add_provisioning_no_mutation
    ON public.research_lab_source_add_provisioning_events;
CREATE TRIGGER trg_research_lab_source_add_provisioning_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_add_provisioning_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_add_provisioning_mutation();

COMMENT ON COLUMN public.research_lab_source_add_submissions.source_identity_hash IS
    'Private duplicate-detection hash for SOURCE_ADD api/docs/domain identity.';

COMMENT ON COLUMN public.research_lab_source_catalog.source_identity_hash IS
    'Private duplicate-detection hash carried from accepted SOURCE_ADD submission.';

COMMENT ON TABLE public.research_lab_source_add_provisioning_events IS
    'Append-only owner provisioning events for SOURCE_ADD APIs eligible for autoresearch.';
