-- Permit validated SOURCE_ADD auth metadata in durable V2 catalog replay.
--
-- The generic execution-result check rejects the word "authorization"
-- anywhere in result_doc. Authenticated SOURCE_ADD catalogs legitimately use
-- Authorization as the name of a header whose value stays encrypted and is
-- injected only inside the coordinator. Strip only that exact, correlated
-- metadata field for the secret-pattern scan; every other occurrence remains
-- forbidden.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION
public.research_lab_attested_execution_result_secret_free_v2(
    p_operation TEXT,
    p_result_doc JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $$
DECLARE
    secret_pattern CONSTANT TEXT :=
        '(sk-or-|sb_secret|service_role|openrouter_api_key|'
        'scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|'
        'private_repo|judge_prompt|hidden_icp|provider_output|request_body|'
        'response_body|authorization|proxy-authorization|'
        '://[^/]+:[^/@]+@)';
    projected JSONB := p_result_doc;
    sources JSONB;
    routes JSONB;
    source_item JSONB;
    provider_item JSONB;
    route_item JSONB;
    source_index INTEGER;
    route_index INTEGER;
    match_count INTEGER;
BEGIN
    IF jsonb_typeof(p_result_doc) <> 'object' THEN
        RETURN FALSE;
    END IF;
    IF p_operation <> 'source_add_catalog_snapshot_v2' THEN
        RETURN p_result_doc::TEXT !~* secret_pattern;
    END IF;

    IF p_result_doc->>'schema_version'
           <> 'leadpoet.source_add_catalog_snapshot.v2'
       OR NOT p_result_doc ?& ARRAY[
            'schema_version',
            'provisioned_sources',
            'provisioned_sources_hash',
            'private_registry_rows',
            'private_registry_rows_hash',
            'runtime_catalog',
            'runtime_catalog_hash'
       ]
       OR p_result_doc - 'schema_version' - 'provisioned_sources'
            - 'provisioned_sources_hash' - 'private_registry_rows'
            - 'private_registry_rows_hash' - 'runtime_catalog'
            - 'runtime_catalog_hash' <> '{}'::JSONB
       OR jsonb_typeof(p_result_doc->'provisioned_sources') <> 'array'
       OR jsonb_typeof(p_result_doc->'private_registry_rows') <> 'array'
       OR jsonb_typeof(p_result_doc->'runtime_catalog') <> 'object'
       OR jsonb_typeof(p_result_doc->'runtime_catalog'->'routes') <> 'array'
    THEN
        RETURN FALSE;
    END IF;

    sources := p_result_doc->'provisioned_sources';
    routes := p_result_doc->'runtime_catalog'->'routes';

    FOR source_index IN 0..jsonb_array_length(sources) - 1 LOOP
        source_item := sources->source_index;
        provider_item :=
            source_item->'provision_doc'->'provider_registry_entry';
        IF jsonb_typeof(source_item) <> 'object'
           OR jsonb_typeof(provider_item) <> 'object'
        THEN
            RETURN FALSE;
        END IF;
        IF lower(COALESCE(provider_item->>'auth_name', '')) = 'authorization'
        THEN
            IF lower(COALESCE(provider_item->>'auth_kind', ''))
                   NOT IN ('header', 'bearer')
               OR COALESCE(provider_item->>'id', '') = ''
            THEN
                RETURN FALSE;
            END IF;
            SELECT count(*)
              INTO match_count
              FROM jsonb_array_elements(routes) AS route(value)
             WHERE jsonb_typeof(route.value) = 'object'
               AND route.value->>'provider_id' = provider_item->>'id'
               AND lower(COALESCE(route.value->>'auth_kind', '')) =
                   lower(provider_item->>'auth_kind')
               AND lower(COALESCE(route.value->>'auth_name', '')) =
                   'authorization';
            IF match_count <> 1 THEN
                RETURN FALSE;
            END IF;
            projected := jsonb_set(
                projected,
                ARRAY[
                    'provisioned_sources', source_index::TEXT,
                    'provision_doc', 'provider_registry_entry'
                ],
                provider_item - 'auth_name',
                FALSE
            );
        END IF;
    END LOOP;

    FOR route_index IN 0..jsonb_array_length(routes) - 1 LOOP
        route_item := routes->route_index;
        IF jsonb_typeof(route_item) <> 'object' THEN
            RETURN FALSE;
        END IF;
        IF lower(COALESCE(route_item->>'auth_name', '')) = 'authorization'
        THEN
            IF lower(COALESCE(route_item->>'auth_kind', ''))
                   NOT IN ('header', 'bearer')
               OR COALESCE(route_item->>'provider_id', '') = ''
            THEN
                RETURN FALSE;
            END IF;
            SELECT count(*)
              INTO match_count
              FROM jsonb_array_elements(sources) AS source(value)
             WHERE jsonb_typeof(source.value) = 'object'
               AND source.value->'provision_doc'->'provider_registry_entry'
                       ->>'id' = route_item->>'provider_id'
               AND lower(COALESCE(
                       source.value->'provision_doc'->'provider_registry_entry'
                           ->>'auth_kind',
                       ''
                   )) = lower(route_item->>'auth_kind')
               AND lower(COALESCE(
                       source.value->'provision_doc'->'provider_registry_entry'
                           ->>'auth_name',
                       ''
                   )) = 'authorization';
            IF match_count <> 1 THEN
                RETURN FALSE;
            END IF;
            projected := jsonb_set(
                projected,
                ARRAY['runtime_catalog', 'routes', route_index::TEXT],
                route_item - 'auth_name',
                FALSE
            );
        END IF;
    END LOOP;

    RETURN projected::TEXT !~* secret_pattern;
EXCEPTION
    WHEN OTHERS THEN
        RETURN FALSE;
END;
$$;

ALTER TABLE public.research_lab_attested_execution_results_v2
    DROP CONSTRAINT IF EXISTS
        research_lab_attested_execution_results_v2_result_doc_check;
ALTER TABLE public.research_lab_attested_execution_results_v2
    ADD CONSTRAINT
        research_lab_attested_execution_results_v2_result_doc_check
    CHECK (
        public.research_lab_attested_execution_result_secret_free_v2(
            operation,
            result_doc
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_attested_execution_results_v2
    VALIDATE CONSTRAINT
        research_lab_attested_execution_results_v2_result_doc_check;

REVOKE ALL ON FUNCTION
    public.research_lab_attested_execution_result_secret_free_v2(TEXT, JSONB)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_attested_execution_result_secret_free_v2(TEXT, JSONB)
    TO service_role;

COMMENT ON FUNCTION
    public.research_lab_attested_execution_result_secret_free_v2(TEXT, JSONB)
IS
    'Fail-closed secret scan for replayable V2 results; permits only correlated SOURCE_ADD Authorization auth_name metadata.';

NOTIFY pgrst, 'reload schema';

COMMIT;
