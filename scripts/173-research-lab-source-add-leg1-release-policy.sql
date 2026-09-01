-- Set the production SOURCE_ADD Leg 1 release policy to 0.2% with a
-- global FIFO admission cap of 50 approvals per UTC day.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Keep the old gateway from admitting 1% / 10-cap work while the database
-- contract advances ahead of the exact candidate restart.
DO $$
BEGIN
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF NOT COALESCE((
        SELECT paused
        FROM public.research_lab_source_add_control
        WHERE singleton
    ), FALSE) THEN
        RAISE EXCEPTION 'SOURCE_ADD must be paused before Leg 1 policy migration';
    END IF;
    LOCK TABLE public.research_lab_source_add_work_items
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items
        WHERE work_status = 'leased'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD work is leased during Leg 1 policy migration';
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_reserve_leg1_slot_v3(
    p_intent_id TEXT,
    p_work_id TEXT,
    p_work_lease_token UUID,
    p_daily_cap INTEGER,
    p_slot_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_oldest_work_id TEXT;
    v_day DATE := (NOW() AT TIME ZONE 'UTC')::DATE;
    v_retry_at TIMESTAMPTZ := NOW() + INTERVAL '5 seconds';
BEGIN
    IF p_slot_lease_seconds < 30 OR p_slot_lease_seconds > 1800 THEN
        RAISE EXCEPTION 'SOURCE_ADD reward slot policy is invalid';
    END IF;
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND OR v_work.work_status <> 'leased'
       OR v_work.work_kind <> 'leg1_reward'
       OR v_work.lease_token IS DISTINCT FROM p_work_lease_token THEN
        RETURN pg_catalog.jsonb_build_object('status', 'lease_lost');
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE intent_id = p_intent_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('status', 'intent_missing');
    END IF;
    IF v_work.submission_id <> v_intent.submission_id
       OR v_work.adapter_id <> v_intent.adapter_id
       OR v_work.job_doc->>'intent_id' <> p_intent_id THEN
        RAISE EXCEPTION 'SOURCE_ADD reward intent scope differs';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-leg1-day:' || v_day::TEXT, 0)
    );
    -- A response-loss retry owns its existing live reservation. Refresh that
    -- exact slot before considering later FIFO changes; demoting the work
    -- while leaving its reserved slot live would consume the daily cap twice.
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_slots existing_slot
        WHERE existing_slot.intent_id = p_intent_id
          AND existing_slot.work_id = p_work_id
          AND existing_slot.slot_day = v_day
          AND existing_slot.slot_status = 'reserved'
          AND existing_slot.lease_expires_at > NOW()
    ) THEN
        RETURN public.research_lab_source_add_reserve_leg1_slot(
            p_intent_id,
            p_work_id,
            p_work_lease_token,
            50,
            p_slot_lease_seconds
        );
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_obligations reward
        WHERE reward.adapter_id = v_intent.adapter_id
          AND reward.leg = 1
    ) THEN
        SELECT candidate.work_id INTO v_oldest_work_id
        FROM public.research_lab_source_add_work_items candidate
        JOIN public.research_lab_source_add_reward_intents candidate_intent
          ON candidate_intent.intent_id = candidate.job_doc->>'intent_id'
         AND candidate_intent.submission_id = candidate.submission_id
         AND candidate_intent.adapter_id = candidate.adapter_id
         AND candidate_intent.leg = 1
        WHERE candidate.work_kind = 'leg1_reward'
          AND (
              candidate.work_status = 'leased'
              OR (
                  candidate.work_status IN ('queued', 'retry_wait')
                  AND candidate.available_at <= NOW()
              )
          )
          AND candidate_intent.intent_status IN ('queued', 'leased', 'retry_wait')
          AND candidate_intent.available_at <= NOW()
          AND NOT EXISTS (
              SELECT 1
              FROM public.research_lab_source_add_reward_obligations existing
              WHERE existing.adapter_id = candidate.adapter_id
                AND existing.leg = 1
          )
        ORDER BY candidate.priority ASC,
                 candidate.available_at ASC,
                 candidate.created_at ASC,
                 candidate.work_id ASC
        LIMIT 1;
        IF v_oldest_work_id IS NOT NULL
           AND v_oldest_work_id <> p_work_id THEN
            UPDATE public.research_lab_source_add_reward_intents
            SET intent_status = 'retry_wait',
                available_at = v_retry_at,
                updated_at = NOW()
            WHERE intent_id = p_intent_id;
            UPDATE public.research_lab_source_add_work_items
            SET work_status = 'retry_wait',
                available_at = v_retry_at,
                lease_token = NULL,
                leased_by = '',
                lease_expires_at = NULL,
                result_doc = pg_catalog.jsonb_build_object(
                    'status', 'fifo_wait'
                ),
                updated_at = NOW()
            WHERE work_id = p_work_id;
            RETURN pg_catalog.jsonb_build_object(
                'status', 'fifo_wait',
                'available_at', v_retry_at
            );
        END IF;
    END IF;
    RETURN public.research_lab_source_add_reserve_leg1_slot(
        p_intent_id,
        p_work_id,
        p_work_lease_token,
        50,
        p_slot_lease_seconds
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_leg1_v3(
    p_intent_id TEXT,
    p_work_id TEXT,
    p_work_lease_token UUID,
    p_slot_lease_token UUID,
    p_daily_cap INTEGER,
    p_reward JSONB,
    p_submission_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_result JSONB;
BEGIN
    IF pg_catalog.jsonb_typeof(p_reward) <> 'object'
       OR p_reward->>'state' <> 'active'
       OR COALESCE((p_reward->>'alpha_percent')::NUMERIC, 0) <> 0.2
       OR COALESCE((p_reward->>'reward_epochs')::INTEGER, 0) <> 20 THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 release economics differs';
    END IF;
    v_result := public.research_lab_source_add_finalize_leg1(
        p_intent_id,
        p_work_id,
        p_work_lease_token,
        p_slot_lease_token,
        50,
        p_reward,
        p_submission_doc
    );
    IF COALESCE(v_result->>'status', '') = 'created'
       AND NOT EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_reward_events event
           WHERE event.reward_ref = v_result->>'reward_ref'
             AND event.seq = 0
             AND event.reward_status = 'active'
             AND event.reason = 'leg1_functional_probe_passed'
       ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 initial reward event differs';
    END IF;
    RETURN v_result;
END;
$$;


REVOKE ALL ON FUNCTION public.research_lab_source_add_reserve_leg1_slot_v3(
    TEXT, TEXT, UUID, INTEGER, INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_leg1_v3(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_reserve_leg1_slot_v3(
    TEXT, TEXT, UUID, INTEGER, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_leg1_v3(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) TO service_role;


CREATE OR REPLACE FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v2()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_service_role_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role'
    ) INTO v_service_role_exists;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.source_add_post_accept_leg1_contract.v2',
        'daily_cap', 50,
        'leg1_alpha_percent', 0.2,
        'leg1_reward_epochs', 20,
        'function_authority_sha256', (
            SELECT 'sha256:' || pg_catalog.encode(
                extensions.digest(
                    pg_catalog.convert_to(
                        COALESCE(
                            pg_catalog.jsonb_object_agg(
                                authority.name,
                                pg_catalog.jsonb_build_object(
                                    'body', proc.prosrc,
                                    'security_definer', proc.prosecdef,
                                    'configuration', pg_catalog.to_jsonb(
                                        proc.proconfig
                                    ),
                                    'language', language.lanname,
                                    'volatility', proc.provolatile,
                                    'parallel', proc.proparallel,
                                    'kind', proc.prokind,
                                    'return_type', proc.prorettype::REGTYPE::TEXT
                                )
                            ),
                            '{}'::JSONB
                        )::TEXT,
                        'UTF8'
                    ),
                    'sha256'
                ),
                'hex'
            )
            FROM (
                VALUES
                    (
                        'claim_work',
                        'public.research_lab_source_add_claim_work(text,integer)'
                    ),
                    (
                        'configure_probe',
                        'public.research_lab_source_add_configure_probe(text,text,jsonb,jsonb,text,text,text)'
                    ),
                    (
                        'configure_probe_v2',
                        'public.research_lab_source_add_configure_probe_v2(text,text,jsonb,jsonb,text,text,text)'
                    ),
                    (
                        'contract_v1',
                        'public.research_lab_source_add_post_accept_leg1_contract_v1()'
                    ),
                    (
                        'contract_v2',
                        'public.research_lab_source_add_post_accept_leg1_contract_v2()'
                    ),
                    (
                        'enqueue_provision_smoke',
                        'public.research_lab_source_add_enqueue_provision_smoke(text,text,text,text,jsonb,jsonb)'
                    ),
                    (
                        'final_approval_catalog_v2',
                        'public.research_lab_source_add_final_approval_catalog_v2(text)'
                    ),
                    (
                        'finalize_leg1',
                        'public.research_lab_source_add_finalize_leg1(text,text,uuid,uuid,integer,jsonb,jsonb)'
                    ),
                    (
                        'finalize_leg1_v2',
                        'public.research_lab_source_add_finalize_leg1_v2(text,text,uuid,uuid,integer,jsonb,jsonb)'
                    ),
                    (
                        'finalize_leg1_v3',
                        'public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)'
                    ),
                    (
                        'finalize_provision',
                        'public.research_lab_source_add_finalize_provision(text,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'finalize_provision_smoke_v2',
                        'public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'finalize_provision_v2',
                        'public.research_lab_source_add_finalize_provision_v2(text,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'finish_work',
                        'public.research_lab_source_add_finish_work(text,uuid,text,text,jsonb,text,jsonb,jsonb,jsonb,jsonb,jsonb,jsonb,timestamp with time zone,boolean)'
                    ),
                    (
                        'reject_current_builtin_v2',
                        'public.research_lab_source_add_reject_current_builtin_v2(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'reserve_leg1_slot',
                        'public.research_lab_source_add_reserve_leg1_slot(text,text,uuid,integer,integer)'
                    ),
                    (
                        'reserve_leg1_slot_v2',
                        'public.research_lab_source_add_reserve_leg1_slot_v2(text,text,uuid,integer,integer)'
                    ),
                    (
                        'reserve_leg1_slot_v3',
                        'public.research_lab_source_add_reserve_leg1_slot_v3(text,text,uuid,integer,integer)'
                    ),
                    (
                        'trigger_acceptance_v2',
                        'public.enforce_research_lab_source_add_acceptance_v2()'
                    ),
                    (
                        'trigger_eligible_v2',
                        'public.enforce_research_lab_source_add_eligible_v2()'
                    ),
                    (
                        'trigger_leg1_initial_event_v2',
                        'public.enforce_research_lab_source_add_leg1_initial_event_v2()'
                    ),
                    (
                        'trigger_leg1_obligation_v2',
                        'public.enforce_research_lab_source_add_leg1_obligation_v2()'
                    ),
                    (
                        'trigger_leg1_slot_v2',
                        'public.enforce_research_lab_source_add_leg1_slot_v2()'
                    ),
                    (
                        'trigger_leg1_work_v2',
                        'public.enforce_research_lab_source_add_leg1_work_v2()'
                    )
            ) AS authority(name, signature)
            LEFT JOIN pg_catalog.pg_proc proc
              ON proc.oid = pg_catalog.to_regprocedure(authority.signature)
            LEFT JOIN pg_catalog.pg_language language
              ON language.oid = proc.prolang
        ),
        'functions', pg_catalog.jsonb_build_object(
            'configure_probe_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_configure_probe_v2(text,text,jsonb,jsonb,text,text,text)'
            ) IS NOT NULL,
            'finalize_provision_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_provision_v2(text,jsonb,jsonb,jsonb)'
            ) IS NOT NULL,
            'reject_current_builtin_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_reject_current_builtin_v2(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)'
            ) IS NOT NULL,
            'post_accept_contract_v1', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_post_accept_leg1_contract_v1()'
            ) IS NOT NULL,
            'reserve_leg1_slot_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_reserve_leg1_slot_v2(text,text,uuid,integer,integer)'
            ) IS NOT NULL,
            'finalize_leg1_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_leg1_v2(text,text,uuid,uuid,integer,jsonb,jsonb)'
            ) IS NOT NULL,
            'reserve_leg1_slot_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_reserve_leg1_slot_v3(text,text,uuid,integer,integer)'
            ) IS NOT NULL,
            'finalize_leg1_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)'
            ) IS NOT NULL,
            'finalize_provision_smoke_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)'
            ) IS NOT NULL
        ),
        'triggers', pg_catalog.jsonb_build_object(
            'acceptance', EXISTS (
                SELECT 1 FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation
                  ON relation.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'research_lab_source_add_submissions'
                  AND trigger_row.tgname = 'trg_source_add_acceptance_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_acceptance_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'eligible', EXISTS (
                SELECT 1 FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation
                  ON relation.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'research_lab_source_add_provisioning_events'
                  AND trigger_row.tgname = 'trg_source_add_eligible_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_eligible_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_work', EXISTS (
                SELECT 1 FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation
                  ON relation.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'research_lab_source_add_work_items'
                  AND trigger_row.tgname = 'trg_source_add_leg1_work_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_work_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_slot', EXISTS (
                SELECT 1 FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation
                  ON relation.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'research_lab_source_add_reward_slots'
                  AND trigger_row.tgname = 'trg_source_add_leg1_slot_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 23
                  AND trigger_row.tgattr::TEXT = (
                      SELECT pg_catalog.string_agg(
                          attribute.attnum::TEXT,
                          ' ' ORDER BY CASE attribute.attname
                              WHEN 'slot_status' THEN 1
                              WHEN 'intent_id' THEN 2
                          END
                      )
                      FROM pg_catalog.pg_attribute attribute
                      WHERE attribute.attrelid = relation.oid
                        AND attribute.attname IN ('slot_status', 'intent_id')
                        AND NOT attribute.attisdropped
                  )
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_slot_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_obligation', EXISTS (
                SELECT 1 FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation
                  ON relation.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'research_lab_source_add_reward_obligations'
                  AND trigger_row.tgname = 'trg_source_add_leg1_obligation_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_obligation_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_initial_event', EXISTS (
                SELECT 1 FROM pg_catalog.pg_trigger trigger_row
                JOIN pg_catalog.pg_class relation
                  ON relation.oid = trigger_row.tgrelid
                JOIN pg_catalog.pg_namespace namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = 'research_lab_source_add_reward_events'
                  AND trigger_row.tgname = 'trg_source_add_leg1_initial_event_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_initial_event_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            )
        ),
        'permissions', pg_catalog.jsonb_build_object(
            'service_role_exists', v_service_role_exists,
            'candidate_callable', v_service_role_exists AND
                pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_configure_probe_v2(text,text,jsonb,jsonb,text,text,text)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_v2(text,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reject_current_builtin_v2(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reserve_leg1_slot_v3(text,text,uuid,integer,integer)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                ),
            'rollback_v2_callable', v_service_role_exists AND
                pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reserve_leg1_slot_v2(text,text,uuid,integer,integer)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_leg1_v2(text,text,uuid,uuid,integer,jsonb,jsonb)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_post_accept_leg1_contract_v1()',
                    'EXECUTE'
                ),
            'legacy_not_callable', v_service_role_exists AND NOT (
                pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_configure_probe(text,text,jsonb,jsonb,text,text,text)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision(text,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reserve_leg1_slot(text,text,uuid,integer,integer)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_leg1(text,text,uuid,uuid,integer,jsonb,jsonb)',
                    'EXECUTE'
                ) OR pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_smoke(text,uuid,text,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
            )
        )
    );
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v2()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v2()
    TO service_role;

COMMENT ON TABLE public.research_lab_source_add_reward_obligations IS
    'Append-only SOURCE_ADD reward legs: each finally accepted source may create its own 0.2% Leg 1 obligation; enabled implementation riders are separate obligations. Active percentages sum deterministically up to the Research Lab cap.';

NOTIFY pgrst, 'reload schema';

COMMIT;
