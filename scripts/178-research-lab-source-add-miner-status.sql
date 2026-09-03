-- Private miner status for SOURCE_ADD submissions.
--
-- Miners authenticate with Bittensor hotkey signatures, not Supabase JWTs.
-- This projection and its page function therefore remain service-role-only;
-- the gateway verifies the signer and supplies the only ownership filter.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE VIEW public.research_lab_source_add_miner_status_v1
WITH (security_invoker = true, security_barrier = true) AS
WITH raw_status AS (
    SELECT
        current.submission_id,
        current.adapter_id,
        current.miner_hotkey,
        LEFT(
            COALESCE(
                NULLIF(BTRIM(current.submission_doc #>> '{manifest,source_name}'), ''),
                'API source'
            ),
            160
        ) AS source_name,
        first_submission.created_at AS submitted_at,
        GREATEST(
            current.created_at,
            COALESCE(provenance_work.updated_at, current.created_at),
            COALESCE(intent.updated_at, current.created_at),
            COALESCE(
                reward.current_status_at,
                reward.created_at,
                current.created_at
            )
        ) AS updated_at,
        current.stage,
        provenance_work.work_status AS provenance_work_status,
        (
            CASE
                WHEN jsonb_typeof(current.precheck_doc->'reasons') = 'array'
                    THEN current.precheck_doc->'reasons'
                ELSE '[]'::JSONB
            END
            || CASE
                WHEN jsonb_typeof(current.precheck_doc->'reason_codes') = 'array'
                    THEN current.precheck_doc->'reason_codes'
                ELSE '[]'::JSONB
            END
            || CASE
                WHEN jsonb_typeof(
                    current.submission_doc #> '{provenance_result,reasons}'
                ) = 'array'
                    THEN current.submission_doc #> '{provenance_result,reasons}'
                ELSE '[]'::JSONB
            END
        ) AS private_reason_codes,
        authority.submission_id IS NOT NULL AS provenance_approved,
        intent.intent_status,
        reward.reward_ref,
        reward.current_reward_status,
        reward.alpha_percent,
        reward.reward_epochs,
        reward.start_epoch
    FROM public.research_lab_source_add_submission_current current
    JOIN LATERAL (
        SELECT history.created_at
        FROM public.research_lab_source_add_submissions history
        WHERE history.submission_id = current.submission_id
        ORDER BY
            history.seq ASC,
            history.created_at ASC,
            history.submission_row_id ASC
        LIMIT 1
    ) first_submission ON TRUE
    LEFT JOIN LATERAL (
        SELECT work.work_status, work.updated_at
        FROM public.research_lab_source_add_work_items work
        WHERE work.submission_id = current.submission_id
          AND work.work_kind = 'provenance'
        ORDER BY work.created_at DESC, work.work_id DESC
        LIMIT 1
    ) provenance_work ON TRUE
    LEFT JOIN public.research_lab_source_add_provenance_leg1_authority_v1 authority
      ON authority.submission_id = current.submission_id
     AND authority.adapter_id = current.adapter_id
     AND authority.miner_hotkey = current.miner_hotkey
    LEFT JOIN public.research_lab_source_add_reward_intents intent
      ON intent.submission_id = current.submission_id
     AND intent.adapter_id = current.adapter_id
     AND intent.miner_hotkey = current.miner_hotkey
     AND intent.leg = 1
    LEFT JOIN public.research_lab_source_add_reward_current reward
      ON reward.adapter_id = current.adapter_id
     AND reward.miner_hotkey = current.miner_hotkey
     AND reward.leg = 1
), decisions AS (
    SELECT
        raw_status.*,
        (
            raw_status.reward_ref IS NOT NULL
            OR raw_status.provenance_approved
            OR raw_status.intent_status IN ('queued', 'leased', 'retry_wait', 'finalized')
            OR raw_status.stage = 'accepted'
        ) AS is_approved
    FROM raw_status
), classified AS (
    SELECT
        decisions.*,
        CASE
            WHEN decisions.is_approved THEN 'approved'
            WHEN decisions.stage IN (
                'rejected',
                'rejected_precheck',
                'functional_probe_failed'
            ) THEN 'rejected'
            ELSE 'pending'
        END AS decision_status
    FROM decisions
)
SELECT
    'leadpoet.source_add_miner_status.v1'::TEXT AS schema_version,
    classified.submission_id,
    classified.miner_hotkey,
    classified.source_name,
    classified.submitted_at,
    classified.updated_at,
    classified.decision_status,
    CASE
        WHEN classified.decision_status = 'approved'
             AND classified.current_reward_status IN ('active', 'partially_paid')
            THEN 'leg1_reward_active'
        WHEN classified.decision_status = 'approved'
             AND classified.current_reward_status = 'stopped_forward'
            THEN 'leg1_reward_stopped'
        WHEN classified.decision_status = 'approved'
            THEN 'leg1_reward_pending'
        WHEN classified.decision_status = 'pending'
             AND classified.stage = 'needs_manual_review'
             AND COALESCE(classified.provenance_work_status, '') NOT IN (
                 'queued', 'leased', 'retry_wait'
             )
            THEN 'additional_review_needed'
        WHEN classified.decision_status = 'pending'
            THEN 'automated_checks_in_progress'
        WHEN classified.private_reason_codes ?| ARRAY[
            'documentation_contains_fake_or_test_markers',
            'ai_mode_flagged_fake_or_test_api'
        ]
            THEN 'source_credibility_not_verified'
        WHEN classified.private_reason_codes ?| ARRAY[
            'missing_api_base_url',
            'missing_documentation_url',
            'missing_auth_type',
            'missing_rate_limit_notes',
            'missing_endpoint_examples'
        ]
            THEN 'submission_details_not_verified'
        WHEN classified.private_reason_codes ?| ARRAY[
            'docs_domain_not_related_to_api_domain',
            'documentation_fetch_failed',
            'documentation_provider_error',
            'documentation_or_entity_evidence_incomplete'
        ]
            THEN 'documentation_not_verified'
        WHEN classified.private_reason_codes ?| ARRAY[
            'archive_provider_error',
            'ai_mode_no_references',
            'ai_mode_legitimacy_not_confirmed',
            'insufficient_reference_or_archive_provenance'
        ]
            THEN 'provenance_not_verified'
        WHEN classified.stage = 'functional_probe_failed'
            THEN 'technical_validation_not_passed'
        ELSE 'automated_checks_not_passed'
    END AS decision_reason_code,
    CASE
        WHEN classified.decision_status = 'approved'
             AND classified.current_reward_status IN ('active', 'partially_paid')
            THEN 'The source passed automated checks and the Leg 1 reward is active.'
        WHEN classified.decision_status = 'approved'
             AND classified.current_reward_status = 'stopped_forward'
            THEN 'The source passed automated checks. Future Leg 1 reward payments have stopped.'
        WHEN classified.decision_status = 'approved'
            THEN 'The source passed automated checks. Leg 1 reward setup is in progress.'
        WHEN classified.decision_status = 'pending'
             AND classified.stage = 'needs_manual_review'
             AND COALESCE(classified.provenance_work_status, '') NOT IN (
                 'queued', 'leased', 'retry_wait'
             )
            THEN 'Automated verification was inconclusive and needs additional review.'
        WHEN classified.decision_status = 'pending'
            THEN 'Automated Source Add checks are still in progress.'
        WHEN classified.private_reason_codes ?| ARRAY[
            'documentation_contains_fake_or_test_markers',
            'ai_mode_flagged_fake_or_test_api'
        ]
            THEN 'The source did not pass the public credibility checks.'
        WHEN classified.private_reason_codes ?| ARRAY[
            'missing_api_base_url',
            'missing_documentation_url',
            'missing_auth_type',
            'missing_rate_limit_notes',
            'missing_endpoint_examples'
        ]
            THEN 'The submitted API details were incomplete or could not be verified.'
        WHEN classified.private_reason_codes ?| ARRAY[
            'docs_domain_not_related_to_api_domain',
            'documentation_fetch_failed',
            'documentation_provider_error',
            'documentation_or_entity_evidence_incomplete'
        ]
            THEN 'The public API documentation could not be verified.'
        WHEN classified.private_reason_codes ?| ARRAY[
            'archive_provider_error',
            'ai_mode_no_references',
            'ai_mode_legitimacy_not_confirmed',
            'insufficient_reference_or_archive_provenance'
        ]
            THEN 'Independent public evidence for the source could not be verified.'
        WHEN classified.stage = 'functional_probe_failed'
            THEN 'The source did not pass technical validation.'
        ELSE 'The submission did not pass automated Source Add checks.'
    END AS decision_reason,
    CASE
        WHEN classified.decision_status = 'rejected' THEN 'not_eligible'
        WHEN classified.decision_status = 'pending' THEN 'not_decided'
        WHEN classified.current_reward_status IN ('active', 'partially_paid')
            THEN 'active'
        WHEN classified.current_reward_status = 'stopped_forward'
            THEN 'stopped'
        ELSE 'pending'
    END AS reward_status,
    classified.alpha_percent,
    classified.reward_epochs,
    classified.start_epoch,
    CASE
        WHEN classified.start_epoch IS NOT NULL
         AND classified.reward_epochs IS NOT NULL
            THEN classified.start_epoch + classified.reward_epochs - 1
        ELSE NULL
    END AS end_epoch
FROM classified;

REVOKE ALL ON TABLE public.research_lab_source_add_miner_status_v1
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT ON TABLE public.research_lab_source_add_miner_status_v1
    TO service_role;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_miner_status_page_v1(
    p_miner_hotkey TEXT,
    p_cursor_submission_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 20
)
RETURNS SETOF public.research_lab_source_add_miner_status_v1
LANGUAGE sql
STABLE
SECURITY INVOKER
SET search_path = ''
AS $$
    WITH cursor_row AS (
        SELECT status.submitted_at, status.submission_id
        FROM public.research_lab_source_add_miner_status_v1 status
        WHERE status.miner_hotkey = p_miner_hotkey
          AND status.submission_id = p_cursor_submission_id
    )
    SELECT status.*
    FROM public.research_lab_source_add_miner_status_v1 status
    WHERE status.miner_hotkey = p_miner_hotkey
      AND (
          p_cursor_submission_id IS NULL
          OR (
              EXISTS (SELECT 1 FROM cursor_row)
              AND (status.submitted_at, status.submission_id) < (
                  SELECT cursor_row.submitted_at, cursor_row.submission_id
                  FROM cursor_row
              )
          )
      )
    ORDER BY status.submitted_at DESC, status.submission_id DESC
    LIMIT LEAST(GREATEST(COALESCE(p_limit, 20), 1), 50) + 1;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_miner_status_contract_v1()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $function$
DECLARE
    v_view_oid OID := pg_catalog.to_regclass(
        'public.research_lab_source_add_miner_status_v1'
    );
    v_page_oid OID := pg_catalog.to_regprocedure(
        'public.research_lab_source_add_miner_status_page_v1(text,text,integer)'
    );
    v_contract_oid OID := pg_catalog.to_regprocedure(
        'public.research_lab_source_add_miner_status_contract_v1()'
    );
BEGIN
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_miner_status_contract.v1',
        'view_name', 'research_lab_source_add_miner_status_v1',
        'page_rpc', 'research_lab_source_add_miner_status_page_v1',
        'page_signature', 'text,text,integer',
        'view_columns', COALESCE((
            SELECT pg_catalog.jsonb_agg(
                attribute.attname ORDER BY attribute.attnum
            )
            FROM pg_catalog.pg_attribute attribute
            WHERE attribute.attrelid = v_view_oid
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
        ), '[]'::JSONB),
        'view_security_invoker', COALESCE((
            SELECT class.reloptions @> ARRAY['security_invoker=true']
            FROM pg_catalog.pg_class class
            WHERE class.oid = v_view_oid
              AND class.relkind = 'v'
        ), FALSE),
        'view_security_barrier', COALESCE((
            SELECT class.reloptions @> ARRAY['security_barrier=true']
            FROM pg_catalog.pg_class class
            WHERE class.oid = v_view_oid
              AND class.relkind = 'v'
        ), FALSE),
        'page_security_invoker', COALESCE((
            SELECT NOT procedure.prosecdef
            FROM pg_catalog.pg_proc procedure
            WHERE procedure.oid = v_page_oid
        ), FALSE),
        'page_stable', COALESCE((
            SELECT procedure.provolatile = 's'
            FROM pg_catalog.pg_proc procedure
            WHERE procedure.oid = v_page_oid
        ), FALSE),
        'view_authority_sha256', COALESCE((
            SELECT 'sha256:' || pg_catalog.encode(
                extensions.digest(
                    pg_catalog.convert_to(
                        pg_catalog.jsonb_build_object(
                            'definition', pg_catalog.pg_get_viewdef(
                                class.oid, TRUE
                            ),
                            'columns', COALESCE((
                                SELECT pg_catalog.jsonb_agg(
                                    pg_catalog.jsonb_build_object(
                                        'name', attribute.attname,
                                        'type', pg_catalog.format_type(
                                            attribute.atttypid,
                                            attribute.atttypmod
                                        )
                                    ) ORDER BY attribute.attnum
                                )
                                FROM pg_catalog.pg_attribute attribute
                                WHERE attribute.attrelid = class.oid
                                  AND attribute.attnum > 0
                                  AND NOT attribute.attisdropped
                            ), '[]'::JSONB),
                            'kind', class.relkind
                        )::TEXT,
                        'UTF8'
                    ),
                    'sha256'
                ),
                'hex'
            )
            FROM pg_catalog.pg_class class
            WHERE class.oid = v_view_oid
              AND class.relkind = 'v'
        ), ''),
        'page_authority_sha256', COALESCE((
            SELECT 'sha256:' || pg_catalog.encode(
                extensions.digest(
                    pg_catalog.convert_to(
                        pg_catalog.jsonb_build_object(
                            'body', procedure.prosrc,
                            'security_definer', procedure.prosecdef,
                            'configuration', pg_catalog.to_jsonb(
                                procedure.proconfig
                            ),
                            'identity_arguments',
                                pg_catalog.pg_get_function_identity_arguments(
                                    procedure.oid
                                ),
                            'argument_names', pg_catalog.to_jsonb(
                                procedure.proargnames
                            ),
                            'language', language.lanname,
                            'volatility', procedure.provolatile,
                            'parallel', procedure.proparallel,
                            'kind', procedure.prokind,
                            'return_type',
                                procedure.prorettype::REGTYPE::TEXT
                        )::TEXT,
                        'UTF8'
                    ),
                    'sha256'
                ),
                'hex'
            )
            FROM pg_catalog.pg_proc procedure
            JOIN pg_catalog.pg_language language
              ON language.oid = procedure.prolang
            WHERE procedure.oid = v_page_oid
        ), ''),
        'contract_authority_sha256', COALESCE((
            SELECT 'sha256:' || pg_catalog.encode(
                extensions.digest(
                    pg_catalog.convert_to(
                        pg_catalog.jsonb_build_object(
                            'body', procedure.prosrc,
                            'security_definer', procedure.prosecdef,
                            'configuration', pg_catalog.to_jsonb(
                                procedure.proconfig
                            ),
                            'identity_arguments',
                                pg_catalog.pg_get_function_identity_arguments(
                                    procedure.oid
                                ),
                            'argument_names', pg_catalog.to_jsonb(
                                procedure.proargnames
                            ),
                            'language', language.lanname,
                            'volatility', procedure.provolatile,
                            'parallel', procedure.proparallel,
                            'kind', procedure.prokind,
                            'return_type',
                                procedure.prorettype::REGTYPE::TEXT
                        )::TEXT,
                        'UTF8'
                    ),
                    'sha256'
                ),
                'hex'
            )
            FROM pg_catalog.pg_proc procedure
            JOIN pg_catalog.pg_language language
              ON language.oid = procedure.prolang
            WHERE procedure.oid = v_contract_oid
        ), ''),
        'permissions', pg_catalog.jsonb_build_object(
            'view_service_role_select', pg_catalog.has_table_privilege(
                'service_role',
                'public.research_lab_source_add_miner_status_v1',
                'SELECT'
            ),
            'view_anon_select', pg_catalog.has_table_privilege(
                'anon',
                'public.research_lab_source_add_miner_status_v1',
                'SELECT'
            ),
            'view_authenticated_select', pg_catalog.has_table_privilege(
                'authenticated',
                'public.research_lab_source_add_miner_status_v1',
                'SELECT'
            ),
            'view_public_select', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class class
                CROSS JOIN LATERAL pg_catalog.aclexplode(
                    COALESCE(
                        class.relacl,
                        pg_catalog.acldefault('r', class.relowner)
                    )
                ) privilege
                WHERE class.oid = v_view_oid
                  AND privilege.grantee = 0
                  AND privilege.privilege_type = 'SELECT'
            ),
            'page_service_role_callable', pg_catalog.has_function_privilege(
                'service_role',
                'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
                'EXECUTE'
            ),
            'page_anon_callable', pg_catalog.has_function_privilege(
                'anon',
                'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
                'EXECUTE'
            ),
            'page_authenticated_callable', pg_catalog.has_function_privilege(
                'authenticated',
                'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
                'EXECUTE'
            ),
            'page_public_callable', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_proc procedure
                CROSS JOIN LATERAL pg_catalog.aclexplode(
                    COALESCE(
                        procedure.proacl,
                        pg_catalog.acldefault('f', procedure.proowner)
                    )
                ) privilege
                WHERE procedure.oid = v_page_oid
                  AND privilege.grantee = 0
                  AND privilege.privilege_type = 'EXECUTE'
            ),
            'contract_service_role_callable',
                pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_miner_status_contract_v1()',
                    'EXECUTE'
                ),
            'contract_anon_callable', pg_catalog.has_function_privilege(
                'anon',
                'public.research_lab_source_add_miner_status_contract_v1()',
                'EXECUTE'
            ),
            'contract_authenticated_callable',
                pg_catalog.has_function_privilege(
                    'authenticated',
                    'public.research_lab_source_add_miner_status_contract_v1()',
                    'EXECUTE'
                )
        )
    );
END;
$function$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_miner_status_page_v1(TEXT, TEXT, INTEGER)
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_miner_status_page_v1(TEXT, TEXT, INTEGER)
    TO service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_source_add_miner_status_contract_v1()
    FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_miner_status_contract_v1()
    TO service_role;

DO $source_add_miner_status_acl_readback$
BEGIN
    IF NOT pg_catalog.has_table_privilege(
        'service_role',
        'public.research_lab_source_add_miner_status_v1',
        'SELECT'
    ) OR pg_catalog.has_table_privilege(
        'anon',
        'public.research_lab_source_add_miner_status_v1',
        'SELECT'
    ) OR pg_catalog.has_table_privilege(
        'authenticated',
        'public.research_lab_source_add_miner_status_v1',
        'SELECT'
    ) OR NOT pg_catalog.has_function_privilege(
        'service_role',
        'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'anon',
        'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'authenticated',
        'public.research_lab_source_add_miner_status_page_v1(text,text,integer)',
        'EXECUTE'
    ) OR NOT pg_catalog.has_function_privilege(
        'service_role',
        'public.research_lab_source_add_miner_status_contract_v1()',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'anon',
        'public.research_lab_source_add_miner_status_contract_v1()',
        'EXECUTE'
    ) OR pg_catalog.has_function_privilege(
        'authenticated',
        'public.research_lab_source_add_miner_status_contract_v1()',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD miner status ACL readback failed';
    END IF;
END;
$source_add_miner_status_acl_readback$;

NOTIFY pgrst, 'reload schema';

COMMIT;
