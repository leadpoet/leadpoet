-- Create SOURCE_ADD Leg 1 only after accepted, smoke-tested provisioning.
--
-- Historical stopped-forward rewards remain immutable. The triggers below
-- apply to new intents/work/rewards, while the V2 finalizer atomically turns a
-- successful provisioning smoke into exactly one queued Leg 1 decision.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_intent_after_acceptance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_functional_probe_current functional
        JOIN public.research_lab_source_add_provisioning_current provision
          ON provision.submission_id = functional.submission_id
         AND provision.adapter_id = functional.adapter_id
        JOIN public.research_lab_source_add_provisioning_smoke_current smoke
          ON smoke.submission_id = provision.submission_id
         AND smoke.adapter_id = provision.adapter_id
        JOIN public.research_lab_source_catalog catalog
          ON catalog.catalog_id = provision.catalog_id
         AND catalog.adapter_id = provision.adapter_id
        WHERE functional.submission_id = NEW.submission_id
          AND functional.adapter_id = NEW.adapter_id
          AND functional.result_status = 'passed'
          AND functional.receipt_hash = NEW.functional_receipt_hash
          AND functional.business_artifact_hash = NEW.business_artifact_hash
          AND smoke.result_status = 'passed'
          AND smoke.evaluation_mode = 'provisioning_smoke'
          AND smoke.config_ref = functional.config_ref
          AND provision.miner_hotkey = NEW.miner_hotkey
          AND provision.provision_status = 'provisioned_autoresearch_eligible'
          AND catalog.miner_ref = NEW.miner_hotkey
          AND catalog.registry_provider_id = provision.registry_provider_id
          AND EXISTS (
              SELECT 1
              FROM public.research_lab_source_add_submissions accepted
              WHERE accepted.submission_id = NEW.submission_id
                AND accepted.adapter_id = NEW.adapter_id
                AND accepted.miner_hotkey = NEW.miner_hotkey
                AND accepted.stage = 'accepted'
                AND accepted.precheck_status = 'provenance_precheck_passed'
          )
          AND EXISTS (
              SELECT 1
              FROM public.research_lab_attested_execution_receipts_v2 receipt
              JOIN public.research_lab_attested_business_artifact_links_v2 link
                ON link.receipt_hash = receipt.receipt_hash
              WHERE receipt.receipt_hash = functional.receipt_hash
                AND receipt.role = 'gateway_coordinator'
                AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
                AND receipt.receipt_status = 'succeeded'
                AND receipt.output_root = functional.business_artifact_hash
                AND link.artifact_kind = 'source_add_functional_probe'
                AND link.artifact_ref = functional.attempt_ref
                AND link.artifact_hash = functional.business_artifact_hash
          )
          AND EXISTS (
              SELECT 1
              FROM public.research_lab_attested_execution_receipts_v2 receipt
              JOIN public.research_lab_attested_business_artifact_links_v2 link
                ON link.receipt_hash = receipt.receipt_hash
              WHERE receipt.receipt_hash = smoke.receipt_hash
                AND receipt.role = 'gateway_coordinator'
                AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
                AND receipt.receipt_status = 'succeeded'
                AND receipt.output_root = smoke.business_artifact_hash
                AND link.artifact_kind = 'source_add_provisioning_smoke'
                AND link.artifact_ref = smoke.attempt_ref
                AND link.artifact_hash = smoke.business_artifact_hash
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 requires accepted eligible provisioning'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_source_add_leg1_intent_after_acceptance
    ON public.research_lab_source_add_reward_intents;
CREATE TRIGGER trg_source_add_leg1_intent_after_acceptance
    BEFORE INSERT ON public.research_lab_source_add_reward_intents
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_intent_after_acceptance();

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_work_after_acceptance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NEW.work_kind = 'leg1_reward' AND NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents intent
        JOIN public.research_lab_source_add_functional_probe_current functional
          ON functional.submission_id = intent.submission_id
         AND functional.adapter_id = intent.adapter_id
        JOIN public.research_lab_source_add_provisioning_current provision
          ON provision.submission_id = intent.submission_id
         AND provision.adapter_id = intent.adapter_id
         AND provision.miner_hotkey = intent.miner_hotkey
        JOIN public.research_lab_source_add_provisioning_smoke_current smoke
          ON smoke.submission_id = provision.submission_id
         AND smoke.adapter_id = provision.adapter_id
        WHERE intent.intent_id = NEW.job_doc->>'intent_id'
          AND intent.submission_id = NEW.submission_id
          AND intent.adapter_id = NEW.adapter_id
          AND functional.result_status = 'passed'
          AND functional.receipt_hash = intent.functional_receipt_hash
          AND functional.business_artifact_hash = intent.business_artifact_hash
          AND NEW.job_doc = jsonb_build_object(
              'intent_id', intent.intent_id,
              'attempt_ref', functional.attempt_ref
          )
          AND provision.provision_status = 'provisioned_autoresearch_eligible'
          AND smoke.result_status = 'passed'
          AND smoke.evaluation_mode = 'provisioning_smoke'
          AND EXISTS (
              SELECT 1
              FROM public.research_lab_source_add_submissions accepted
              WHERE accepted.submission_id = intent.submission_id
                AND accepted.adapter_id = intent.adapter_id
                AND accepted.miner_hotkey = intent.miner_hotkey
                AND accepted.stage = 'accepted'
                AND accepted.precheck_status = 'provenance_precheck_passed'
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 work requires accepted eligible provisioning'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_source_add_leg1_work_after_acceptance
    ON public.research_lab_source_add_work_items;
CREATE TRIGGER trg_source_add_leg1_work_after_acceptance
    BEFORE INSERT ON public.research_lab_source_add_work_items
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_work_after_acceptance();

CREATE OR REPLACE FUNCTION public.bind_research_lab_source_add_leg1_catalog()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_catalog_id TEXT;
BEGIN
    IF NEW.leg <> 1 THEN
        RETURN NEW;
    END IF;
    SELECT provision.catalog_id INTO v_catalog_id
    FROM public.research_lab_source_add_reward_intents intent
    JOIN public.research_lab_source_add_functional_probe_current functional
      ON functional.submission_id = intent.submission_id
     AND functional.adapter_id = intent.adapter_id
    JOIN public.research_lab_source_add_provisioning_current provision
      ON provision.submission_id = intent.submission_id
     AND provision.adapter_id = intent.adapter_id
     AND provision.miner_hotkey = intent.miner_hotkey
    JOIN public.research_lab_source_add_provisioning_smoke_current smoke
      ON smoke.submission_id = provision.submission_id
     AND smoke.adapter_id = provision.adapter_id
    JOIN public.research_lab_source_catalog catalog
      ON catalog.catalog_id = provision.catalog_id
     AND catalog.adapter_id = provision.adapter_id
     AND catalog.miner_ref = provision.miner_hotkey
    WHERE intent.adapter_id = NEW.adapter_id
      AND intent.miner_hotkey = NEW.miner_hotkey
      AND intent.intent_status = 'leased'
      AND functional.result_status = 'passed'
      AND functional.receipt_hash = intent.functional_receipt_hash
      AND functional.business_artifact_hash = intent.business_artifact_hash
      AND smoke.result_status = 'passed'
      AND smoke.evaluation_mode = 'provisioning_smoke'
      AND smoke.config_ref = functional.config_ref
      AND provision.provision_status = 'provisioned_autoresearch_eligible'
      AND catalog.registry_provider_id = provision.registry_provider_id
      AND NEW.trigger_evidence_doc->>'functional_probe_passed' = 'true'
      AND NEW.trigger_evidence_doc->>'attempt_ref' = functional.attempt_ref
      AND NEW.trigger_evidence_doc->>'functional_probe_receipt_hash' = functional.receipt_hash
      AND NEW.trigger_evidence_doc->>'business_artifact_hash' = functional.business_artifact_hash
      AND NEW.trigger_evidence_doc->>'functional_probe_result_hash' = functional.business_artifact_hash
      AND NEW.trigger_evidence_doc->>'evaluator_version' = functional.result_doc->>'evaluator_version'
      AND NEW.trigger_evidence_doc->>'route_hash' = functional.route_hash
      AND NEW.trigger_evidence_doc->>'provisioning_smoke_passed' = 'true'
      AND NEW.trigger_evidence_doc->>'provisioning_smoke_attempt_ref' = smoke.attempt_ref
      AND NEW.trigger_evidence_doc->>'provisioning_smoke_receipt_hash' = smoke.receipt_hash
      AND NEW.trigger_evidence_doc->>'provisioning_smoke_business_artifact_hash' = smoke.business_artifact_hash
      AND NEW.trigger_evidence_doc->>'provisioning_smoke_result_hash' = smoke.business_artifact_hash
      AND NEW.trigger_evidence_doc->>'submission_id' = intent.submission_id
      AND NEW.trigger_evidence_doc->>'final_acceptance_stage' = 'accepted'
      AND NEW.trigger_evidence_doc->>'provision_ref' = provision.provision_ref
      AND NEW.trigger_evidence_doc->>'catalog_id' = provision.catalog_id
      AND NEW.trigger_evidence_doc->>'registry_provider_id' = provision.registry_provider_id
      AND NEW.trigger_evidence_doc->>'provision_status' = 'provisioned_autoresearch_eligible'
      AND EXISTS (
          SELECT 1
          FROM public.research_lab_source_add_submissions accepted
          WHERE accepted.submission_id = intent.submission_id
            AND accepted.adapter_id = intent.adapter_id
            AND accepted.miner_hotkey = intent.miner_hotkey
            AND accepted.stage = 'accepted'
            AND accepted.precheck_status = 'provenance_precheck_passed'
      )
      AND EXISTS (
          SELECT 1
          FROM public.research_lab_attested_execution_receipts_v2 decision
          JOIN public.research_lab_attested_business_artifact_links_v2 link
            ON link.receipt_hash = decision.receipt_hash
          WHERE decision.role = 'gateway_coordinator'
            AND decision.purpose = 'research_lab.reward_decision.v2'
            AND decision.receipt_status = 'succeeded'
            AND link.artifact_kind = 'source_add_reward_decision'
            AND link.artifact_ref = NEW.reward_ref
            AND link.artifact_hash = decision.output_root
            AND pg_catalog.jsonb_typeof(decision.receipt_doc->'parent_receipt_hashes') = 'array'
            AND pg_catalog.jsonb_array_length(decision.receipt_doc->'parent_receipt_hashes') = 2
            AND decision.receipt_doc->'parent_receipt_hashes' ? functional.receipt_hash
            AND decision.receipt_doc->'parent_receipt_hashes' ? smoke.receipt_hash
            AND (SELECT COUNT(*)
                 FROM public.research_lab_attested_receipt_edges_v2 edge
                 WHERE edge.child_receipt_hash = decision.receipt_hash) = 2
            AND EXISTS (
                SELECT 1
                FROM public.research_lab_attested_receipt_edges_v2 edge
                WHERE edge.child_receipt_hash = decision.receipt_hash
                  AND edge.parent_receipt_hash = functional.receipt_hash
            )
            AND EXISTS (
                SELECT 1
                FROM public.research_lab_attested_receipt_edges_v2 edge
                WHERE edge.child_receipt_hash = decision.receipt_hash
                  AND edge.parent_receipt_hash = smoke.receipt_hash
            )
      );
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 approval or receipt graph differs'
            USING ERRCODE = '55000';
    END IF;
    IF NEW.catalog_id IS NOT NULL AND NEW.catalog_id <> v_catalog_id THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 catalog binding differs'
            USING ERRCODE = '55000';
    END IF;
    NEW.catalog_id := v_catalog_id;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_source_add_leg1_catalog_binding
    ON public.research_lab_source_add_reward_obligations;
CREATE TRIGGER trg_source_add_leg1_catalog_binding
    BEFORE INSERT ON public.research_lab_source_add_reward_obligations
    FOR EACH ROW EXECUTE FUNCTION public.bind_research_lab_source_add_leg1_catalog();

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    p_work_id TEXT,
    p_lease_token UUID,
    p_submission_id TEXT,
    p_catalog_row JSONB,
    p_provision_row JSONB,
    p_smoke_attempt JSONB,
    p_reward_intent JSONB,
    p_next_work JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_result JSONB;
    v_current RECORD;
    v_probe public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_next public.research_lab_source_add_work_items%ROWTYPE;
    v_seq INTEGER;
BEGIN
    IF COALESCE(jsonb_typeof(p_reward_intent), '') <> 'object'
       OR COALESCE(jsonb_typeof(p_next_work), '') <> 'object'
       OR p_reward_intent->>'intent_id' !~ '^source_add_reward_intent:[0-9a-f]{16}$'
       OR p_reward_intent->>'functional_receipt_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reward_intent->>'business_artifact_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR btrim(COALESCE(p_reward_intent->>'miner_hotkey', '')) = ''
       OR p_reward_intent <> jsonb_build_object(
           'intent_id', p_reward_intent->>'intent_id',
           'miner_hotkey', p_reward_intent->>'miner_hotkey',
           'functional_receipt_hash', p_reward_intent->>'functional_receipt_hash',
           'business_artifact_hash', p_reward_intent->>'business_artifact_hash'
       )
       OR p_next_work->>'work_id' !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_next_work->>'work_kind' <> 'leg1_reward'
       OR p_next_work->>'priority' <> '30'
       OR p_next_work <> jsonb_build_object(
           'work_id', p_next_work->>'work_id',
           'work_kind', 'leg1_reward',
           'priority', 30,
           'job_doc', jsonb_build_object(
               'intent_id', p_reward_intent->>'intent_id',
               'attempt_ref', p_next_work->'job_doc'->>'attempt_ref'
           )
       )
       OR p_next_work->'job_doc'->>'attempt_ref' !~ '^source_add_probe_attempt:[0-9a-f]{16}$' THEN
        RAISE EXCEPTION 'SOURCE_ADD post-acceptance Leg 1 input is invalid';
    END IF;
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND OR v_work.work_status <> 'leased'
       OR v_work.work_kind <> 'provisioning_smoke'
       OR v_work.lease_token IS DISTINCT FROM p_lease_token THEN
        RETURN jsonb_build_object('status', 'lease_lost');
    END IF;
    IF v_work.submission_id <> p_submission_id
       OR p_smoke_attempt->>'work_id' <> p_work_id
       OR COALESCE((p_smoke_attempt->>'attempt_number')::INTEGER, 0)
          <> v_work.attempt_count THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning smoke lease binding differs';
    END IF;
    v_result := public.research_lab_source_add_finalize_provision(
        p_submission_id,
        p_catalog_row,
        p_provision_row,
        p_smoke_attempt
    );
    IF v_result->>'status' IN ('provisioned', 'already_provisioned') THEN
        SELECT * INTO v_current
        FROM public.research_lab_source_add_submission_current
        WHERE submission_id = p_submission_id;
        SELECT * INTO v_probe
        FROM public.research_lab_source_add_functional_probe_current
        WHERE submission_id = p_submission_id;
        IF v_current.submission_id IS NULL
           OR v_current.stage <> 'accepted'
           OR v_current.adapter_id <> v_work.adapter_id
           OR v_current.miner_hotkey <> p_reward_intent->>'miner_hotkey'
           OR v_probe.attempt_ref IS NULL
           OR v_probe.adapter_id <> v_work.adapter_id
           OR v_probe.result_status <> 'passed'
           OR v_probe.receipt_hash <> p_reward_intent->>'functional_receipt_hash'
           OR v_probe.business_artifact_hash <> p_reward_intent->>'business_artifact_hash'
           OR v_probe.attempt_ref <> p_next_work->'job_doc'->>'attempt_ref' THEN
            RAISE EXCEPTION 'SOURCE_ADD post-acceptance Leg 1 proof differs';
        END IF;

        INSERT INTO public.research_lab_source_add_reward_intents (
            intent_id, submission_id, adapter_id, miner_hotkey, intent_status,
            functional_receipt_hash, business_artifact_hash
        ) VALUES (
            p_reward_intent->>'intent_id', p_submission_id, v_work.adapter_id,
            p_reward_intent->>'miner_hotkey', 'queued',
            p_reward_intent->>'functional_receipt_hash',
            p_reward_intent->>'business_artifact_hash'
        ) ON CONFLICT (adapter_id, leg) DO NOTHING;
        SELECT * INTO v_intent
        FROM public.research_lab_source_add_reward_intents
        WHERE adapter_id = v_work.adapter_id AND leg = 1;
        IF NOT FOUND OR v_intent.intent_id <> p_reward_intent->>'intent_id'
           OR v_intent.submission_id <> p_submission_id
           OR v_intent.miner_hotkey <> p_reward_intent->>'miner_hotkey'
           OR v_intent.functional_receipt_hash <> p_reward_intent->>'functional_receipt_hash'
           OR v_intent.business_artifact_hash <> p_reward_intent->>'business_artifact_hash' THEN
            RAISE EXCEPTION 'SOURCE_ADD post-acceptance reward intent idempotency differs';
        END IF;

        SELECT * INTO v_next
        FROM public.research_lab_source_add_work_items
        WHERE submission_id = p_submission_id AND work_kind = 'leg1_reward';
        IF FOUND THEN
            IF v_next.work_id <> p_next_work->>'work_id'
               OR v_next.adapter_id <> v_work.adapter_id
               OR v_next.job_doc <> p_next_work->'job_doc' THEN
                RAISE EXCEPTION 'SOURCE_ADD post-acceptance work idempotency differs';
            END IF;
        ELSE
            INSERT INTO public.research_lab_source_add_work_items (
                work_id, submission_id, adapter_id, work_kind, work_status,
                priority, job_doc
            ) VALUES (
                p_next_work->>'work_id', p_submission_id, v_work.adapter_id,
                'leg1_reward', 'queued', 30, p_next_work->'job_doc'
            );
        END IF;

        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
        FROM public.research_lab_source_add_submissions
        WHERE submission_id = p_submission_id;
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
            precheck_status, precheck_doc, source_identity_hash,
            source_identity_version
        ) VALUES (
            p_submission_id, v_current.adapter_id, v_current.miner_hotkey,
            'leg1_queued', v_seq,
            v_current.submission_doc || jsonb_build_object('stage', 'leg1_queued'),
            v_current.precheck_status, v_current.precheck_doc,
            v_current.source_identity_hash, v_current.source_identity_version
        );
        v_result := v_result || jsonb_build_object(
            'leg1_status', 'queued',
            'intent_id', p_reward_intent->>'intent_id',
            'work_id', p_next_work->>'work_id'
        );
    END IF;
    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'completed',
        result_doc = v_result,
        completed_at = NOW(),
        lease_token = NULL,
        leased_by = '',
        lease_expires_at = NULL,
        job_doc = job_doc
            - 'provider_execution_state'
            - 'provider_execution_attempt'
            - 'provider_execution_started_at'
            - 'provider_execution_recovery',
        updated_at = NOW()
    WHERE work_id = p_work_id;
    RETURN v_result;
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT jsonb_build_object(
        'schema_version', 'leadpoet.source_add_post_accept_leg1_contract.v1',
        'intent_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            WHERE trigger.tgrelid = 'public.research_lab_source_add_reward_intents'::REGCLASS
              AND trigger.tgname = 'trg_source_add_leg1_intent_after_acceptance'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'work_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            WHERE trigger.tgrelid = 'public.research_lab_source_add_work_items'::REGCLASS
              AND trigger.tgname = 'trg_source_add_leg1_work_after_acceptance'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'reward_trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            WHERE trigger.tgrelid = 'public.research_lab_source_add_reward_obligations'::REGCLASS
              AND trigger.tgname = 'trg_source_add_leg1_catalog_binding'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'finalizer_present', pg_catalog.to_regprocedure(
            'public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)'
        ) IS NOT NULL
    );
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_leg1_intent_after_acceptance()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_leg1_work_after_acceptance()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.bind_research_lab_source_add_leg1_catalog()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1()
    TO service_role;

COMMENT ON FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
) IS 'Atomically accepts eligible SOURCE_ADD provisioning and queues receipt-bound Leg 1.';
COMMENT ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1() IS
    'Read-only release contract for post-acceptance SOURCE_ADD Leg 1 enforcement.';

NOTIFY pgrst, 'reload schema';

COMMIT;
