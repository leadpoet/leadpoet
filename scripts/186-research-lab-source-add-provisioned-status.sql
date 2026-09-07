-- 186-research-lab-source-add-provisioned-status.sql
-- Rename the active SOURCE_ADD eligible state after retiring the Leg 2 judge.
-- Provisioning events are append-only and their provision_ref/provenance
-- evidence must remain byte-for-byte stable. The base event rows therefore
-- retain their historical literal; the current projection and explicit Leg 1
-- predicates accept both literals and expose the canonical active value.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $source_add_status_table$
BEGIN
    IF to_regclass('public.research_lab_source_add_provisioning_events') IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning events table is missing';
    END IF;
END;
$source_add_status_table$;

-- Keep both the immutable historical literal and the canonical active literal
-- valid. This is permanent compatibility for old append-only rows, not a
-- second active state: the current projection below normalizes the old value.
DO $source_add_status_drop_check$
DECLARE
    constraint_row RECORD;
BEGIN
    FOR constraint_row IN
        SELECT c.conname
          FROM pg_catalog.pg_constraint c
         WHERE c.conrelid =
               'public.research_lab_source_add_provisioning_events'::REGCLASS
           AND c.contype = 'c'
           AND c.conname = ANY (ARRAY[
               -- PostgreSQL truncates the inline constraint name from
               -- migration 78 to this stable 63-byte identifier.
               'research_lab_source_add_provisioning_eve_provision_status_check',
               'research_lab_source_provision_status_check'
           ])
    LOOP
        EXECUTE pg_catalog.format(
            'ALTER TABLE public.research_lab_source_add_provisioning_events DROP CONSTRAINT %I',
            constraint_row.conname
        );
    END LOOP;
END;
$source_add_status_drop_check$;
ALTER TABLE public.research_lab_source_add_provisioning_events
    ADD CONSTRAINT research_lab_source_provision_status_check
    CHECK (provision_status IN (
        'approved_pending_provision', 'provisioned',
        'provisioned_autoresearch_eligible', 'disabled'
    )) NOT VALID;

-- Normalize only the active projection. The append-only event, provision_ref,
-- provision_doc, catalog_doc, credential envelope, and all receipt material
-- remain unchanged.
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
    CASE
        WHEN p.provision_status = 'provisioned_autoresearch_eligible'
            THEN 'provisioned'
        ELSE p.provision_status
    END AS provision_status,
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

-- Support only the deployed Leg 1 function identities below. Do not scan or
-- rewrite arbitrary public functions: their bodies may be unrelated
-- compatibility or settlement authorities.
CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_provision_status_is_eligible_v1(
        p_status TEXT
    )
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $$
    SELECT p_status IN ('provisioned', 'provisioned_autoresearch_eligible');
$$;

DO $source_add_status_functions$
DECLARE
    function_row RECORD;
    function_definition TEXT;
    matched_function_count INTEGER := 0;
    supported_functions CONSTANT TEXT[] := ARRAY[
        'public.research_lab_source_add_final_approval_catalog_v2(text)',
        'public.enforce_research_lab_source_add_acceptance_v2()',
        'public.enforce_research_lab_source_add_eligible_v2()',
        'public.enforce_research_lab_source_add_leg1_obligation_v2()',
        'public.research_lab_source_add_enqueue_provision_smoke(text,text,text,text,jsonb,jsonb)',
        'public.enforce_research_lab_source_add_eligible_v3()',
        'public.research_lab_source_add_enqueue_provision_smoke_v2(text,text,text,text,jsonb,jsonb)',
        'public.research_lab_source_add_finalize_provision_v3(text,jsonb,jsonb,jsonb)',
        'public.research_lab_source_add_reject_current_builtin_v3(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)'
    ];
BEGIN
    FOR function_row IN
        SELECT p.oid
          FROM pg_catalog.pg_proc p
         WHERE p.oid = ANY (
             SELECT pg_catalog.to_regprocedure(function_signature)
             FROM pg_catalog.unnest(supported_functions) AS signatures(function_signature)
         )
    LOOP
        matched_function_count := matched_function_count + 1;
        function_definition := pg_catalog.pg_get_functiondef(function_row.oid);
        -- Function formatting changes between PostgreSQL versions. Match only
        -- the known predicate shapes, while retaining the explicit function
        -- identity allow-list above.
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'provision[.]provision_status[[:space:]]*=[[:space:]]*''provisioned_autoresearch_eligible''',
            'public.research_lab_source_add_provision_status_is_eligible_v1(provision.provision_status)',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'eligible[.]provision_status[[:space:]]*=[[:space:]]*''provisioned_autoresearch_eligible''',
            'public.research_lab_source_add_provision_status_is_eligible_v1(eligible.provision_status)',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            '([[:space:]])provision_status[[:space:]]*=[[:space:]]*''provisioned_autoresearch_eligible''',
            '\1public.research_lab_source_add_provision_status_is_eligible_v1(provision_status)',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'v_provision[.]provision_status[[:space:]]*<>[[:space:]]*''provisioned_autoresearch_eligible''',
            'NOT public.research_lab_source_add_provision_status_is_eligible_v1(v_provision.provision_status)',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'NEW[.]provision_status[[:space:]]*<>[[:space:]]*''provisioned_autoresearch_eligible''',
            'NOT public.research_lab_source_add_provision_status_is_eligible_v1(NEW.provision_status)',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'p_provision_row[[:space:]]*-[[:space:]]*>>[[:space:]]*''provision_status''[[:space:]]*<>[[:space:]]*''provisioned_autoresearch_eligible''',
            'NOT public.research_lab_source_add_provision_status_is_eligible_v1(p_provision_row->>''provision_status'')',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'p_provision_row[[:space:]]*-[[:space:]]*>>[[:space:]]*''provision_status''[[:space:]]*=[[:space:]]*''provisioned_autoresearch_eligible''',
            'public.research_lab_source_add_provision_status_is_eligible_v1(p_provision_row->>''provision_status'')',
            'g'
        );
        function_definition := pg_catalog.regexp_replace(
            function_definition,
            'v_work[.]job_doc[[:space:]]*#[[:space:]]*>[[:space:]]*''[{]provision_row,provision_status[}]''[[:space:]]+IS[[:space:]]+DISTINCT[[:space:]]+FROM[[:space:]]+pg_catalog[.]to_jsonb[[:space:]]*[(][[:space:]]*''provisioned_autoresearch_eligible''::TEXT[[:space:]]*[)]',
            'public.research_lab_source_add_provision_status_is_eligible_v1(v_work.job_doc#>>''{provision_row,provision_status}'') IS NOT TRUE',
            'g'
        );
        IF function_definition ILIKE '%provisioned_autoresearch_eligible%'
           OR function_definition NOT ILIKE
              '%research_lab_source_add_provision_status_is_eligible_v1%'
        THEN
            RAISE EXCEPTION
                'SOURCE_ADD status contract drift in %',
                function_row.oid::REGPROCEDURE;
        END IF;
        EXECUTE function_definition;
    END LOOP;
    IF matched_function_count <> pg_catalog.cardinality(supported_functions) THEN
        RAISE EXCEPTION
            'SOURCE_ADD status function allow-list drift: expected %, found %',
            pg_catalog.cardinality(supported_functions), matched_function_count;
    END IF;
END;
$source_add_status_functions$;

ALTER TABLE public.research_lab_source_add_provisioning_events
    VALIDATE CONSTRAINT research_lab_source_provision_status_check;

COMMIT;
