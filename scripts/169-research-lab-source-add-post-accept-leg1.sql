-- Make SOURCE_ADD Leg 1 contingent on final, smoke-tested acceptance.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- Reward receipts hash RFC-8259 JSON with bytewise-sorted object keys. Keep a
-- SOURCE_ADD-local copy of that small canonicalizer so this migration remains
-- dependency-minimal while independently selecting the exact signed retry
-- attempt which matches the row being created.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_canonical_jsonb_v2(
    p_value JSONB
)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $canonical_json$
BEGIN
    CASE pg_catalog.jsonb_typeof(p_value)
        WHEN 'object' THEN
            RETURN (
                SELECT '{' || coalesce(
                    pg_catalog.string_agg(
                        pg_catalog.to_jsonb(entry.key)::TEXT || ':' ||
                        public.research_lab_source_add_canonical_jsonb_v2(entry.value),
                        ',' ORDER BY entry.key COLLATE "C"
                    ), ''
                ) || '}'
                FROM pg_catalog.jsonb_each(p_value) AS entry(key, value)
            );
        WHEN 'array' THEN
            RETURN (
                SELECT '[' || coalesce(
                    pg_catalog.string_agg(
                        public.research_lab_source_add_canonical_jsonb_v2(entry.value),
                        ',' ORDER BY entry.ordinality
                    ), ''
                ) || ']'
                FROM pg_catalog.jsonb_array_elements(p_value)
                    WITH ORDINALITY AS entry(value, ordinality)
            );
        ELSE
            RETURN p_value::TEXT;
    END CASE;
END;
$canonical_json$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_jsonb_hash_v2(
    p_value JSONB
)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
STRICT
SET search_path = pg_catalog, public
AS $jsonb_hash$
    SELECT 'sha256:' || pg_catalog.encode(
        extensions.digest(
            pg_catalog.convert_to(
                public.research_lab_source_add_canonical_jsonb_v2(p_value),
                'UTF8'
            ),
            'sha256'
        ),
        'hex'
    )
$jsonb_hash$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_canonical_jsonb_v2(JSONB)
    FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION public.research_lab_source_add_jsonb_hash_v2(JSONB)
    FROM PUBLIC, anon, authenticated, service_role;

-- This migration intentionally makes the N-1 functional-pass reward path
-- fail closed. Apply it only while SOURCE_ADD is paused and no affected work
-- lease is in flight, then start the candidate gateway before unpausing.
DO $$
BEGIN
    -- A claim which already read the old unpaused control state retains an
    -- ACCESS SHARE lock on this table until its transaction ends. Fail and
    -- retry instead of letting that claim cross the migration boundary. A
    -- claim which has not read control yet waits until this transaction
    -- commits and then observes the still-paused row.
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF NOT COALESCE((
        SELECT paused
        FROM public.research_lab_source_add_control
        WHERE singleton
    ), FALSE) THEN
        RAISE EXCEPTION 'SOURCE_ADD must be paused before post-accept Leg 1 migration';
    END IF;
    -- SOURCE_ADD writers do not have one universal table order (legacy
    -- admission writes submissions before work, while worker finalizers start
    -- from work). Acquire every affected write/trigger target up front and
    -- never wait: any in-flight writer makes this migration roll back for a
    -- clean retry while paused.
    LOCK TABLE
        public.research_lab_source_add_work_items,
        public.research_lab_source_add_submissions,
        public.research_lab_source_add_functional_probe_attempts,
        public.research_lab_source_catalog,
        public.research_lab_source_add_provisioning_events,
        public.research_lab_source_add_reward_intents,
        public.research_lab_source_add_reward_slots,
        public.research_lab_source_add_reward_obligations,
        public.research_lab_source_add_reward_events
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items
        WHERE work_status = 'leased'
          AND work_kind IN ('functional_probe', 'provisioning_smoke', 'leg1_reward')
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD affected work is leased during post-accept Leg 1 migration';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_events terminal
        JOIN public.research_lab_source_add_reward_events later
          ON later.reward_ref = terminal.reward_ref
         AND later.seq > terminal.seq
        JOIN public.research_lab_source_add_reward_obligations reward
          ON reward.reward_ref = terminal.reward_ref
         AND reward.leg = 1
        WHERE terminal.reward_status = 'stopped_forward'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 terminal history requires adjudication';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submissions accepted
        WHERE accepted.stage = 'accepted'
          AND NOT EXISTS (
              SELECT 1
              FROM public.research_lab_source_add_provisioning_events provision
              JOIN public.research_lab_source_add_functional_probe_attempts smoke
                ON smoke.submission_id = provision.submission_id
               AND smoke.adapter_id = provision.adapter_id
               AND smoke.evaluation_mode = 'provisioning_smoke'
               AND smoke.result_status = 'passed'
              WHERE provision.submission_id = accepted.submission_id
                AND provision.adapter_id = accepted.adapter_id
                AND provision.miner_hotkey = accepted.miner_hotkey
                AND provision.provision_status = 'provisioned_autoresearch_eligible'
                AND provision.created_at <= accepted.created_at
                AND smoke.created_at <= accepted.created_at
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD has a pre-final acceptance requiring adjudication';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents intent
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_provisioning_events provision
            JOIN public.research_lab_source_add_functional_probe_attempts smoke
              ON smoke.submission_id = provision.submission_id
             AND smoke.adapter_id = provision.adapter_id
             AND smoke.evaluation_mode = 'provisioning_smoke'
             AND smoke.result_status = 'passed'
            WHERE provision.submission_id = intent.submission_id
              AND provision.adapter_id = intent.adapter_id
              AND provision.miner_hotkey = intent.miner_hotkey
              AND provision.provision_status = 'provisioned_autoresearch_eligible'
              AND EXISTS (
                  SELECT 1
                  FROM public.research_lab_source_add_submissions accepted
                  WHERE accepted.submission_id = intent.submission_id
                    AND accepted.adapter_id = intent.adapter_id
                    AND accepted.miner_hotkey = intent.miner_hotkey
                    AND accepted.stage = 'accepted'
                    AND provision.created_at <= accepted.created_at
                    AND smoke.created_at <= accepted.created_at
              )
        )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD has a pre-accept Leg 1 intent requiring adjudication';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_current reward
        WHERE reward.leg = 1
          -- Preserve terminal legacy obligations as audit history. They are
          -- excluded from allocation; every absent or nonterminal status must
          -- still prove final smoke-tested acceptance below.
          AND reward.current_reward_status IS DISTINCT FROM 'stopped_forward'
          AND NOT EXISTS (
              SELECT 1
              FROM public.research_lab_source_add_provisioning_events provision
              JOIN public.research_lab_source_add_functional_probe_attempts smoke
                ON smoke.submission_id = provision.submission_id
               AND smoke.adapter_id = provision.adapter_id
               AND smoke.evaluation_mode = 'provisioning_smoke'
               AND smoke.result_status = 'passed'
              WHERE provision.adapter_id = reward.adapter_id
                AND provision.miner_hotkey = reward.miner_hotkey
                AND provision.provision_status = 'provisioned_autoresearch_eligible'
                AND reward.catalog_id IS NOT NULL
                AND reward.catalog_id = provision.catalog_id
                AND provision.created_at <= reward.created_at
                AND smoke.created_at <= reward.created_at
                AND EXISTS (
                    SELECT 1
                    FROM public.research_lab_source_add_submissions accepted
                    WHERE accepted.submission_id = provision.submission_id
                      AND accepted.adapter_id = provision.adapter_id
                      AND accepted.miner_hotkey = provision.miner_hotkey
                      AND accepted.stage = 'accepted'
                      AND accepted.created_at <= reward.created_at
                )
          )
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD has a pre-accept Leg 1 obligation requiring adjudication';
    END IF;
END;
$$;

-- Return the accepted catalog only when the intent, original functional proof,
-- provisioning smoke, and append-only acceptance history all agree.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_final_approval_catalog_v2(
    p_intent_id TEXT
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_functional public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_provision RECORD;
BEGIN
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE intent_id = p_intent_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD final approval reward intent is missing';
    END IF;
    SELECT * INTO v_functional
    FROM public.research_lab_source_add_functional_probe_current
    WHERE submission_id = v_intent.submission_id;
    IF NOT FOUND
       OR v_functional.adapter_id <> v_intent.adapter_id
       OR v_functional.result_status <> 'passed'
       OR v_functional.receipt_hash <> v_intent.functional_receipt_hash
       OR v_functional.business_artifact_hash <> v_intent.business_artifact_hash THEN
        RAISE EXCEPTION 'SOURCE_ADD final approval functional proof differs';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        JOIN public.research_lab_attested_business_artifact_links_v2 link
          ON link.receipt_hash = receipt.receipt_hash
        WHERE receipt.receipt_hash = v_functional.receipt_hash
          AND receipt.role = 'gateway_coordinator'
          AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
          AND receipt.receipt_status = 'succeeded'
          AND receipt.output_root = v_functional.business_artifact_hash
          AND link.artifact_kind = 'source_add_functional_probe'
          AND link.artifact_ref = v_functional.attempt_ref
          AND link.artifact_hash = v_functional.business_artifact_hash
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD final approval functional receipt is unavailable';
    END IF;
    SELECT * INTO v_smoke
    FROM public.research_lab_source_add_provisioning_smoke_current
    WHERE submission_id = v_intent.submission_id;
    IF NOT FOUND
       OR v_smoke.adapter_id <> v_intent.adapter_id
       OR v_smoke.result_status <> 'passed'
       OR v_smoke.config_ref <> v_functional.config_ref THEN
        RAISE EXCEPTION 'SOURCE_ADD final approval smoke proof differs';
    END IF;
    SELECT * INTO v_provision
    FROM public.research_lab_source_add_provisioning_current
    WHERE submission_id = v_intent.submission_id
      AND adapter_id = v_intent.adapter_id
      AND miner_hotkey = v_intent.miner_hotkey
      AND provision_status = 'provisioned_autoresearch_eligible';
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD final approval provisioning is missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submissions accepted
        WHERE accepted.submission_id = v_intent.submission_id
          AND accepted.adapter_id = v_intent.adapter_id
          AND accepted.miner_hotkey = v_intent.miner_hotkey
          AND accepted.stage = 'accepted'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD final acceptance is missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        JOIN public.research_lab_attested_business_artifact_links_v2 link
          ON link.receipt_hash = receipt.receipt_hash
        WHERE receipt.receipt_hash = v_smoke.receipt_hash
          AND receipt.role = 'gateway_coordinator'
          AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
          AND receipt.receipt_status = 'succeeded'
          AND receipt.output_root = v_smoke.business_artifact_hash
          AND link.artifact_kind = 'source_add_provisioning_smoke'
          AND link.artifact_ref = v_smoke.attempt_ref
          AND link.artifact_hash = v_smoke.business_artifact_hash
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD final approval smoke receipt is unavailable';
    END IF;
    RETURN v_provision.catalog_id;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_final_approval_catalog_v2(TEXT)
    FROM PUBLIC, anon, authenticated, service_role;

-- Migration 96 used the same condition for pending and eligible provisioning,
-- so its pending call attempted to append accepted. Suppress only that legacy
-- pending transition. Eligible acceptance remains fail-closed below.
CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_acceptance_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_provision RECORD;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
BEGIN
    IF NEW.stage <> 'accepted' THEN
        RETURN NEW;
    END IF;
    SELECT * INTO v_provision
    FROM public.research_lab_source_add_provisioning_current
    WHERE submission_id = NEW.submission_id
      AND adapter_id = NEW.adapter_id;
    IF FOUND AND v_provision.provision_status = 'approved_pending_provision' THEN
        RETURN NULL;
    END IF;
    IF NOT FOUND
       OR v_provision.miner_hotkey <> NEW.miner_hotkey
       OR v_provision.provision_status <> 'provisioned_autoresearch_eligible' THEN
        RAISE EXCEPTION 'SOURCE_ADD acceptance requires eligible provisioning';
    END IF;
    SELECT * INTO v_smoke
    FROM public.research_lab_source_add_provisioning_smoke_current
    WHERE submission_id = NEW.submission_id;
    IF NOT FOUND
       OR v_smoke.adapter_id <> NEW.adapter_id
       OR v_smoke.result_status <> 'passed' THEN
        RAISE EXCEPTION 'SOURCE_ADD acceptance requires a passed provisioning smoke';
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE submission_id = NEW.submission_id
      AND adapter_id = NEW.adapter_id
      AND miner_hotkey = NEW.miner_hotkey
      AND leg = 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD acceptance requires its Leg 1 intent';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_acceptance_v2()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_acceptance_v2
    ON public.research_lab_source_add_submissions;
CREATE TRIGGER trg_source_add_acceptance_v2
    BEFORE INSERT ON public.research_lab_source_add_submissions
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_acceptance_v2();

-- An eligible provisioning event is the final-approval boundary. It may be
-- created only after the original functional intent exists and the exact
-- smoke attempt has already been persisted in this transaction.
CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_eligible_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_functional public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
BEGIN
    IF NEW.provision_status <> 'provisioned_autoresearch_eligible' THEN
        RETURN NEW;
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE submission_id = NEW.submission_id
      AND adapter_id = NEW.adapter_id
      AND miner_hotkey = NEW.miner_hotkey
      AND leg = 1;
    SELECT * INTO v_functional
    FROM public.research_lab_source_add_functional_probe_current
    WHERE submission_id = NEW.submission_id;
    SELECT * INTO v_smoke
    FROM public.research_lab_source_add_provisioning_smoke_current
    WHERE submission_id = NEW.submission_id;
    IF v_intent.intent_id IS NULL
       OR v_functional.attempt_ref IS NULL
       OR v_smoke.attempt_ref IS NULL
       OR v_functional.adapter_id <> NEW.adapter_id
       OR v_functional.result_status <> 'passed'
       OR v_functional.receipt_hash <> v_intent.functional_receipt_hash
       OR v_functional.business_artifact_hash <> v_intent.business_artifact_hash
       OR v_smoke.adapter_id <> NEW.adapter_id
       OR v_smoke.result_status <> 'passed'
       OR v_smoke.config_ref <> v_functional.config_ref
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_execution_receipts_v2 receipt
           JOIN public.research_lab_attested_business_artifact_links_v2 link
             ON link.receipt_hash = receipt.receipt_hash
           WHERE receipt.receipt_hash = v_functional.receipt_hash
             AND receipt.role = 'gateway_coordinator'
             AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
             AND receipt.receipt_status = 'succeeded'
             AND receipt.output_root = v_functional.business_artifact_hash
             AND link.artifact_kind = 'source_add_functional_probe'
             AND link.artifact_ref = v_functional.attempt_ref
             AND link.artifact_hash = v_functional.business_artifact_hash
       )
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_execution_receipts_v2 receipt
           JOIN public.research_lab_attested_business_artifact_links_v2 link
             ON link.receipt_hash = receipt.receipt_hash
           WHERE receipt.receipt_hash = v_smoke.receipt_hash
             AND receipt.role = 'gateway_coordinator'
             AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
             AND receipt.receipt_status = 'succeeded'
             AND receipt.output_root = v_smoke.business_artifact_hash
             AND link.artifact_kind = 'source_add_provisioning_smoke'
             AND link.artifact_ref = v_smoke.attempt_ref
             AND link.artifact_hash = v_smoke.business_artifact_hash
       ) THEN
        RAISE EXCEPTION 'SOURCE_ADD eligible provisioning lacks post-smoke Leg 1 authority';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_eligible_v2()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_eligible_v2
    ON public.research_lab_source_add_provisioning_events;
CREATE TRIGGER trg_source_add_eligible_v2
    BEFORE INSERT ON public.research_lab_source_add_provisioning_events
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_eligible_v2();

-- Leg 1 work, daily-slot reservation, and obligation creation each recheck the
-- final approval independently. The obligation trigger also replaces the old
-- NULL catalog_id with the exact accepted catalog.
CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_work_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent_id TEXT;
BEGIN
    IF NEW.work_kind <> 'leg1_reward' THEN
        RETURN NEW;
    END IF;
    v_intent_id := NEW.job_doc->>'intent_id';
    IF v_intent_id !~ '^source_add_reward_intent:[0-9a-f]{16}$'
       OR public.research_lab_source_add_final_approval_catalog_v2(v_intent_id) = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 work requires final approval';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_leg1_work_v2()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_work_v2
    ON public.research_lab_source_add_work_items;
CREATE TRIGGER trg_source_add_leg1_work_v2
    BEFORE INSERT ON public.research_lab_source_add_work_items
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_work_v2();

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_slot_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NEW.slot_status = 'reserved'
       AND public.research_lab_source_add_final_approval_catalog_v2(NEW.intent_id) = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 slot requires final approval';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_leg1_slot_v2()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_slot_v2
    ON public.research_lab_source_add_reward_slots;
CREATE TRIGGER trg_source_add_leg1_slot_v2
    BEFORE INSERT OR UPDATE OF slot_status, intent_id
    ON public.research_lab_source_add_reward_slots
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_slot_v2();

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_obligation_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_catalog_id TEXT;
    v_functional public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_provision RECORD;
    v_expected_trigger JSONB;
    v_expected_parents JSONB;
    v_expected_projection JSONB;
    v_expected_decision_hash TEXT;
    v_alpha_json JSONB;
    v_decision RECORD;
    v_decision_count INTEGER;
BEGIN
    IF NEW.leg <> 1 THEN
        RETURN NEW;
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE adapter_id = NEW.adapter_id AND leg = 1;
    IF NOT FOUND OR v_intent.miner_hotkey <> NEW.miner_hotkey THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 obligation owner differs';
    END IF;
    v_catalog_id := public.research_lab_source_add_final_approval_catalog_v2(
        v_intent.intent_id
    );
    IF NEW.catalog_id IS NOT NULL AND NEW.catalog_id <> v_catalog_id THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 obligation catalog differs';
    END IF;
    SELECT * INTO v_functional
    FROM public.research_lab_source_add_functional_probe_current
    WHERE submission_id = v_intent.submission_id;
    SELECT * INTO v_smoke
    FROM public.research_lab_source_add_provisioning_smoke_current
    WHERE submission_id = v_intent.submission_id;
    SELECT * INTO v_provision
    FROM public.research_lab_source_add_provisioning_current
    WHERE submission_id = v_intent.submission_id
      AND adapter_id = v_intent.adapter_id
      AND miner_hotkey = v_intent.miner_hotkey
      AND catalog_id = v_catalog_id
      AND provision_status = 'provisioned_autoresearch_eligible';
    IF v_functional.attempt_ref IS NULL
       OR v_smoke.attempt_ref IS NULL
       OR v_provision.provision_ref IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 approval evidence is missing';
    END IF;
    v_expected_trigger := jsonb_build_object(
        'functional_probe_passed', TRUE,
        'attempt_ref', v_functional.attempt_ref,
        'functional_probe_receipt_hash', v_functional.receipt_hash,
        'business_artifact_hash', v_functional.business_artifact_hash,
        'functional_probe_result_hash', v_functional.business_artifact_hash,
        'evaluator_version', v_functional.result_doc->>'evaluator_version',
        'route_hash', v_functional.route_hash,
        'provisioning_smoke_passed', TRUE,
        'provisioning_smoke_attempt_ref', v_smoke.attempt_ref,
        'provisioning_smoke_receipt_hash', v_smoke.receipt_hash,
        'provisioning_smoke_business_artifact_hash', v_smoke.business_artifact_hash,
        'provisioning_smoke_result_hash', v_smoke.business_artifact_hash,
        'submission_id', v_intent.submission_id,
        'final_acceptance_stage', 'accepted',
        'provision_ref', v_provision.provision_ref,
        'catalog_id', v_provision.catalog_id,
        'registry_provider_id', v_provision.registry_provider_id,
        'provision_status', v_provision.provision_status
    );
    IF NEW.trigger_evidence_doc IS DISTINCT FROM v_expected_trigger THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 final approval evidence differs';
    END IF;
    SELECT pg_catalog.jsonb_agg(parent_hash ORDER BY parent_hash)
    INTO v_expected_parents
    FROM (
        VALUES (v_functional.receipt_hash), (v_smoke.receipt_hash)
    ) expected(parent_hash);
    IF v_functional.receipt_hash = v_smoke.receipt_hash THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 approval receipt roots collide';
    END IF;
    -- Python's signed projection casts alpha_percent to float, whose canonical
    -- JSON retains a trailing .0 for whole values. Rebuild that exact token
    -- before hashing instead of inheriting the NUMERIC column's scale.
    v_alpha_json := CASE
        WHEN NEW.alpha_percent = pg_catalog.trunc(NEW.alpha_percent)
        THEN (
            pg_catalog.to_json(NEW.alpha_percent::DOUBLE PRECISION)::TEXT || '.0'
        )::JSONB
        ELSE pg_catalog.to_jsonb(NEW.alpha_percent::DOUBLE PRECISION)
    END;
    v_expected_projection := pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.reward_row_projection.v2',
        'decision_kind', 'source_add_leg1',
        'reward_row', pg_catalog.jsonb_build_object(
            'reward_ref', NEW.reward_ref,
            'adapter_id', NEW.adapter_id,
            'miner_hotkey', NEW.miner_hotkey,
            'leg', NEW.leg,
            'reward_kind', NEW.reward_kind,
            'alpha_percent', v_alpha_json,
            'reward_epochs', NEW.reward_epochs,
            'start_epoch', NEW.start_epoch,
            'initial_reward_status', 'active',
            'trigger_evidence_doc', NEW.trigger_evidence_doc,
            'public_label', NEW.public_label
        )
    );
    v_expected_decision_hash :=
        public.research_lab_source_add_jsonb_hash_v2(v_expected_projection);
    SELECT COUNT(*)::INTEGER
    INTO v_decision_count
    FROM public.research_lab_attested_business_artifact_links_v2 link
    WHERE link.artifact_kind = 'source_add_reward_decision'
      AND link.artifact_ref = NEW.reward_ref
      AND link.artifact_hash = v_expected_decision_hash;
    IF v_decision_count <> 1 THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 exact reward decision is missing or ambiguous';
    END IF;
    SELECT receipt.receipt_hash, receipt.output_root, receipt.receipt_doc,
           link.artifact_hash
    INTO v_decision
    FROM public.research_lab_attested_business_artifact_links_v2 link
    JOIN public.research_lab_attested_execution_receipts_v2 receipt
      ON receipt.receipt_hash = link.receipt_hash
    WHERE link.artifact_kind = 'source_add_reward_decision'
      AND link.artifact_ref = NEW.reward_ref
      AND link.artifact_hash = v_expected_decision_hash
      AND receipt.role = 'gateway_coordinator'
      AND receipt.purpose = 'research_lab.reward_decision.v2'
      AND receipt.receipt_status = 'succeeded';
    IF NOT FOUND
       OR v_decision.output_root <> v_decision.artifact_hash
       OR v_decision.receipt_doc->'parent_receipt_hashes'
          IS DISTINCT FROM v_expected_parents
       OR (SELECT COUNT(*)
           FROM public.research_lab_attested_receipt_edges_v2 edge
           WHERE edge.child_receipt_hash = v_decision.receipt_hash) <> 2
       OR (SELECT COUNT(*)
           FROM public.research_lab_attested_receipt_edges_v2 edge
           WHERE edge.child_receipt_hash = v_decision.receipt_hash
             AND edge.parent_receipt_hash IN (
                 v_functional.receipt_hash, v_smoke.receipt_hash
             )) <> 2 THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 reward decision ancestry differs';
    END IF;
    NEW.catalog_id := v_catalog_id;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_leg1_obligation_v2()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_obligation_v2
    ON public.research_lab_source_add_reward_obligations;
CREATE TRIGGER trg_source_add_leg1_obligation_v2
    BEFORE INSERT ON public.research_lab_source_add_reward_obligations
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_obligation_v2();

-- An operator may explicitly retry a conclusively failed provisioning smoke
-- without changing the deterministic provision/work identity.  The original
-- enqueue function treated every existing row as already queued, including a
-- completed failure that the dispatcher can never claim again.  Preserve the
-- work identity and prior append-only smoke attempt while moving only a safe,
-- terminal failure back to queued for a new attempt number.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_enqueue_provision_smoke(
    p_work_id TEXT,
    p_submission_id TEXT,
    p_config_ref TEXT,
    p_host_hash TEXT,
    p_catalog_row JSONB,
    p_provision_row JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_current RECORD;
    v_probe public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_config public.research_lab_source_add_probe_config_events%ROWTYPE;
    v_provision public.research_lab_source_add_provisioning_events%ROWTYPE;
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_job_doc JSONB;
    v_terminal_result_status TEXT;
    v_terminal_status TEXT;
    v_work_found BOOLEAN;
BEGIN
    IF p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR p_config_ref !~ '^source_add_probe_config:[0-9a-f]{16}$'
       OR p_host_hash !~ '^sha256:[0-9a-f]{64}$'
       OR jsonb_typeof(p_catalog_row) <> 'object'
       OR jsonb_typeof(p_provision_row) <> 'object'
       OR p_provision_row->>'provision_status' <> 'provisioned_autoresearch_eligible'
       OR p_catalog_row::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
       OR p_provision_row::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])' THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning smoke input is invalid';
    END IF;
    -- Serialize an absent/present deterministic work identity before waiting
    -- on its row.  Worker completion locks work before submission, so enqueue
    -- must never hold the submission lock while waiting for this row.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-work:' || p_work_id, 0)
    );
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    v_work_found := FOUND;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-submission:' || p_submission_id, 0)
    );
    SELECT * INTO v_current
    FROM public.research_lab_source_add_submission_current
    WHERE submission_id = p_submission_id;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'missing'); END IF;
    IF v_current.adapter_id <> p_catalog_row->>'adapter_id'
       OR v_current.adapter_id <> p_provision_row->>'adapter_id'
       OR v_current.miner_hotkey <> p_provision_row->>'miner_hotkey' THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning smoke ownership differs';
    END IF;
    SELECT * INTO v_config
    FROM public.research_lab_source_add_probe_config_current
    WHERE submission_id = p_submission_id AND config_status = 'active';
    SELECT * INTO v_probe
    FROM public.research_lab_source_add_functional_probe_current
    WHERE submission_id = p_submission_id;
    IF v_config.config_ref IS NULL OR v_probe.attempt_ref IS NULL
       OR v_config.config_ref <> p_config_ref
       OR v_probe.result_status <> 'passed'
       OR v_probe.config_ref <> p_config_ref THEN
        RETURN jsonb_build_object('status', 'current_probe_config_required');
    END IF;
    SELECT
        provision_event_id,
        provision_ref,
        catalog_id,
        submission_id,
        adapter_id,
        miner_hotkey,
        source_identity_hash,
        registry_provider_id,
        provision_status,
        seq,
        provision_doc,
        credential_envelope,
        created_at
    INTO v_provision
    FROM public.research_lab_source_add_provisioning_current
    WHERE adapter_id = v_current.adapter_id;
    IF v_provision.provision_event_id IS NULL
       OR v_provision.provision_status <> 'approved_pending_provision' THEN
        RETURN jsonb_build_object('status', 'pending_approval_required');
    END IF;

    v_job_doc := jsonb_build_object(
        'config_ref', p_config_ref,
        'host_hash', p_host_hash,
        'catalog_row', p_catalog_row,
        'provision_row', p_provision_row
    );
    IF v_work_found THEN
        IF v_work.submission_id <> p_submission_id
           OR v_work.adapter_id <> v_current.adapter_id
           OR v_work.work_kind <> 'provisioning_smoke'
           OR (v_work.job_doc
               - 'provider_execution_state'
               - 'provider_execution_attempt'
               - 'provider_execution_started_at'
               - 'provider_execution_recovery') <> v_job_doc THEN
            RAISE EXCEPTION 'SOURCE_ADD provisioning smoke idempotency differs';
        END IF;
        IF v_work.work_status = 'completed' THEN
            v_terminal_result_status := COALESCE(
                v_work.result_doc->>'result_status', ''
            );
            v_terminal_status := COALESCE(v_work.result_doc->>'status', '');
            IF v_work.attempt_count < 20
               AND (
                   (
                       v_terminal_result_status IN (
                           'failed', 'manual_review', 'awaiting_operator', 'retryable'
                       )
                       AND EXISTS (
                           SELECT 1
                           FROM public.research_lab_source_add_functional_probe_attempts attempt
                           WHERE attempt.work_id = p_work_id
                             AND attempt.attempt_number = v_work.attempt_count
                             AND attempt.evaluation_mode = 'provisioning_smoke'
                             AND attempt.result_status = v_terminal_result_status
                       )
                   )
                   OR v_terminal_status = 'worker_exception_dead_letter'
                   OR (
                       v_terminal_status = 'current_model_catalog_unavailable'
                       AND EXISTS (
                           SELECT 1
                           FROM public.research_lab_source_add_functional_probe_attempts attempt
                           WHERE attempt.work_id = p_work_id
                             AND attempt.attempt_number = v_work.attempt_count
                             AND attempt.evaluation_mode = 'provisioning_smoke'
                             AND attempt.result_status = 'passed'
                       )
                   )
               )
               AND NOT EXISTS (
                   SELECT 1
                   FROM public.research_lab_source_add_submissions accepted
                   WHERE accepted.submission_id = p_submission_id
                     AND accepted.stage IN ('accepted', 'leg1_queued', 'leg1_created')
               )
               AND NOT EXISTS (
                   SELECT 1
                   FROM public.research_lab_source_add_reward_intents intent
                   WHERE intent.adapter_id = v_work.adapter_id
               )
               AND NOT EXISTS (
                   SELECT 1
                   FROM public.research_lab_source_add_provisioning_events eligible
                   WHERE eligible.adapter_id = v_work.adapter_id
                     AND eligible.provision_status = 'provisioned_autoresearch_eligible'
               ) THEN
                UPDATE public.research_lab_source_add_work_items
                SET work_status = 'queued',
                    available_at = NOW(),
                    lease_token = NULL,
                    leased_by = '',
                    lease_expires_at = NULL,
                    completed_at = NULL,
                    job_doc = v_job_doc,
                    result_doc = jsonb_build_object(
                        'status', 'operator_requeued',
                        'prior_result_status', COALESCE(
                            NULLIF(v_terminal_result_status, ''),
                            v_terminal_status
                        )
                    ),
                    updated_at = NOW()
                WHERE work_id = p_work_id;
                RETURN jsonb_build_object(
                    'status', 'queued',
                    'work_id', p_work_id,
                    'work_status', 'queued',
                    'requeued', TRUE
                );
            END IF;
            RETURN jsonb_build_object(
                'status', 'terminal_retry_not_allowed',
                'work_id', p_work_id,
                'work_status', v_work.work_status
            );
        END IF;
        RETURN jsonb_build_object(
            'status', 'already_queued', 'work_id', p_work_id,
            'work_status', v_work.work_status
        );
    END IF;
    INSERT INTO public.research_lab_source_add_work_items (
        work_id, submission_id, adapter_id, work_kind, work_status, priority,
        job_doc
    ) VALUES (
        p_work_id, p_submission_id, v_current.adapter_id,
        'provisioning_smoke', 'queued', 25, v_job_doc
    );
    RETURN jsonb_build_object(
        'status', 'queued', 'work_id', p_work_id, 'work_status', 'queued'
    );
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke(
    TEXT, TEXT, TEXT, TEXT, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke(
    TEXT, TEXT, TEXT, TEXT, JSONB, JSONB
) TO service_role;

COMMENT ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke(
    TEXT, TEXT, TEXT, TEXT, JSONB, JSONB
) IS 'Queues a deterministic provisioning smoke and permits only explicit, conclusively safe terminal retry without changing work identity.';

-- The candidate-only finalizer is the sole product path that can cross from a
-- passed smoke into accepted/eligible state and queue Leg 1. All writes occur
-- in the caller's single database transaction.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    p_work_id TEXT,
    p_lease_token UUID,
    p_submission_id TEXT,
    p_catalog_row JSONB,
    p_provision_row JSONB,
    p_smoke_attempt JSONB,
    p_reward_intent JSONB,
    p_reward_work JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_current RECORD;
    v_functional public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_reward_work public.research_lab_source_add_work_items%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_expected_job JSONB;
    v_result JSONB;
BEGIN
    IF p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR jsonb_typeof(p_reward_intent) <> 'object'
       OR jsonb_typeof(p_reward_work) <> 'object'
       OR p_reward_intent->>'intent_id' !~ '^source_add_reward_intent:[0-9a-f]{16}$'
       OR p_reward_intent->>'functional_receipt_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reward_intent->>'business_artifact_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reward_work->>'work_id' !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_reward_work->>'work_kind' <> 'leg1_reward'
       OR COALESCE((p_reward_work->>'priority')::INTEGER, 0) <> 30
       OR p_reward_work->'job_doc'->>'intent_id' <> p_reward_intent->>'intent_id'
       OR p_reward_work->'job_doc'->>'attempt_ref' !~ '^source_add_probe_attempt:[0-9a-f]{16}$' THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept Leg 1 input is invalid';
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
    v_expected_job := jsonb_build_object(
        'config_ref', p_smoke_attempt->>'config_ref',
        'host_hash', v_work.job_doc->>'host_hash',
        'catalog_row', p_catalog_row,
        'provision_row', p_provision_row
    );
    IF v_work.submission_id <> p_submission_id
       OR p_smoke_attempt->>'work_id' <> p_work_id
       OR COALESCE((p_smoke_attempt->>'attempt_number')::INTEGER, 0)
          <> v_work.attempt_count
       OR p_smoke_attempt->>'attempt_ref' <>
          'source_add_probe_attempt:' || pg_catalog.substr(
              public.research_lab_source_add_jsonb_hash_v2(
                  pg_catalog.jsonb_build_object(
                      'prefix', 'source_add_probe_attempt',
                      'parts', pg_catalog.jsonb_build_array(
                          p_submission_id, p_work_id, v_work.attempt_count
                      )
                  )
              ),
              8,
              16
          )
       OR (v_work.job_doc
           - 'provider_execution_state'
           - 'provider_execution_attempt'
           - 'provider_execution_started_at'
           - 'provider_execution_recovery') <> v_expected_job THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept smoke lease binding differs';
    END IF;
    SELECT * INTO v_current
    FROM public.research_lab_source_add_submission_current
    WHERE submission_id = p_submission_id;
    SELECT * INTO v_functional
    FROM public.research_lab_source_add_functional_probe_current
    WHERE submission_id = p_submission_id;
    IF v_current.submission_id IS NULL
       OR v_functional.attempt_ref IS NULL
       OR v_current.adapter_id <> v_work.adapter_id
       OR v_current.miner_hotkey <> p_reward_intent->>'miner_hotkey'
       OR v_functional.adapter_id <> v_work.adapter_id
       OR v_functional.result_status <> 'passed'
       OR v_functional.receipt_hash <> p_reward_intent->>'functional_receipt_hash'
       OR v_functional.business_artifact_hash <> p_reward_intent->>'business_artifact_hash'
       OR v_functional.attempt_ref <> p_reward_work->'job_doc'->>'attempt_ref'
       OR v_functional.config_ref <> p_smoke_attempt->>'config_ref' THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept functional authority differs';
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
    IF NOT FOUND
       OR v_intent.intent_id <> p_reward_intent->>'intent_id'
       OR v_intent.submission_id <> p_submission_id
       OR v_intent.miner_hotkey <> p_reward_intent->>'miner_hotkey'
       OR v_intent.functional_receipt_hash <> p_reward_intent->>'functional_receipt_hash'
       OR v_intent.business_artifact_hash <> p_reward_intent->>'business_artifact_hash' THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept reward intent idempotency differs';
    END IF;

    v_result := public.research_lab_source_add_finalize_provision(
        p_submission_id,
        p_catalog_row,
        p_provision_row,
        p_smoke_attempt
    );
    IF COALESCE(v_result->>'status', '') NOT IN ('provisioned', 'already_provisioned') THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept provisioning failed: %',
            COALESCE(v_result->>'status', 'missing');
    END IF;
    SELECT * INTO v_smoke
    FROM public.research_lab_source_add_functional_probe_attempts
    WHERE attempt_ref = p_smoke_attempt->>'attempt_ref';
    IF NOT FOUND
       OR v_smoke.submission_id <> p_submission_id
       OR v_smoke.adapter_id <> v_work.adapter_id
       OR v_smoke.work_id <> p_work_id
       OR v_smoke.attempt_number <> v_work.attempt_count
       OR v_smoke.evaluation_mode <> 'provisioning_smoke'
       OR v_smoke.config_ref <> p_smoke_attempt->>'config_ref'
       OR v_smoke.result_status <> 'passed'
       OR v_smoke.route_hash <> p_smoke_attempt->>'route_hash'
       OR v_smoke.response_hash <> COALESCE(p_smoke_attempt->>'response_hash', '')
       OR v_smoke.status_class <> COALESCE(p_smoke_attempt->>'status_class', '')
       OR v_smoke.content_type <> COALESCE(p_smoke_attempt->>'content_type', '')
       OR v_smoke.byte_count <> COALESCE((p_smoke_attempt->>'byte_count')::INTEGER, 0)
       OR v_smoke.duration_ms <> COALESCE((p_smoke_attempt->>'duration_ms')::INTEGER, 0)
       OR v_smoke.retry_after_seconds <>
          COALESCE((p_smoke_attempt->>'retry_after_seconds')::INTEGER, 0)
       OR v_smoke.reason_codes <> COALESCE(p_smoke_attempt->'reason_codes', '[]'::JSONB)
       OR v_smoke.receipt_hash <> p_smoke_attempt->>'receipt_hash'
       OR v_smoke.business_artifact_hash <>
          p_smoke_attempt->>'business_artifact_hash'
       OR v_smoke.result_doc <> p_smoke_attempt->'result_doc' THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept persisted smoke differs from lease';
    END IF;

    INSERT INTO public.research_lab_source_add_work_items (
        work_id, submission_id, adapter_id, work_kind, work_status,
        priority, job_doc
    ) VALUES (
        p_reward_work->>'work_id', p_submission_id, v_work.adapter_id,
        'leg1_reward', 'queued', 30, p_reward_work->'job_doc'
    ) ON CONFLICT (work_id) DO NOTHING;
    SELECT * INTO v_reward_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_reward_work->>'work_id';
    IF NOT FOUND
       OR v_reward_work.submission_id <> p_submission_id
       OR v_reward_work.adapter_id <> v_work.adapter_id
       OR v_reward_work.work_kind <> 'leg1_reward'
       OR v_reward_work.job_doc <> p_reward_work->'job_doc' THEN
        RAISE EXCEPTION 'SOURCE_ADD post-accept reward work idempotency differs';
    END IF;

    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'completed',
        result_doc = v_result || jsonb_build_object(
            'leg1_intent_id', v_intent.intent_id,
            'leg1_work_id', v_reward_work.work_id
        ),
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
    RETURN v_result || jsonb_build_object(
        'leg1_intent_id', v_intent.intent_id,
        'leg1_work_id', v_reward_work.work_id
    );
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
) TO service_role;

COMMENT ON FUNCTION public.research_lab_source_add_finalize_provision_smoke_v2(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB
) IS 'Atomically persists passed provisioning smoke, final acceptance, and the exact functional-receipt-bound Leg 1 work.';

-- Once final approval exists, configuration and provisioning are frozen until
-- the corresponding Leg 1 obligation is durable.  These V2 entry points keep
-- that invariant in the database even if a stale gateway calls the N-1 RPCs.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_configure_probe_v2(
    p_submission_id TEXT,
    p_config_ref TEXT,
    p_probe_doc JSONB,
    p_credential_envelope JSONB,
    p_actor_ref TEXT,
    p_work_id TEXT,
    p_host_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-submission:' || p_submission_id, 0
        )
    );
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submissions history
        WHERE history.submission_id = p_submission_id
          AND history.stage IN ('accepted', 'leg1_queued', 'leg1_created')
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents intent
        WHERE intent.submission_id = p_submission_id
    ) THEN
        RETURN pg_catalog.jsonb_build_object('status', 'terminal');
    END IF;
    RETURN public.research_lab_source_add_configure_probe(
        p_submission_id,
        p_config_ref,
        p_probe_doc,
        p_credential_envelope,
        p_actor_ref,
        p_work_id,
        p_host_hash
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_provision_v2(
    p_submission_id TEXT,
    p_catalog_row JSONB,
    p_provision_row JSONB,
    p_smoke_attempt JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-submission:' || p_submission_id, 0
        )
    );
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submissions terminal
        WHERE terminal.submission_id = p_submission_id
          AND terminal.stage IN ('accepted', 'leg1_queued', 'leg1_created')
    ) AND NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents intent
        JOIN public.research_lab_source_add_reward_obligations reward
          ON reward.adapter_id = intent.adapter_id
         AND reward.leg = 1
        WHERE intent.submission_id = p_submission_id
          AND intent.leg = 1
          AND intent.intent_status = 'finalized'
          AND intent.reward_ref = reward.reward_ref
    ) THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'final_approval_frozen'
        );
    END IF;
    RETURN public.research_lab_source_add_finalize_provision(
        p_submission_id,
        p_catalog_row,
        p_provision_row,
        p_smoke_attempt
    );
END;
$$;

-- A passed smoke which resolves to a current model provider is terminally
-- ineligible. Persist the exact measured attempt, complete its work, and
-- append a disabled provisioning event atomically so it cannot retain an open
-- quota slot or a misleading pending registry row.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_reject_current_builtin_v2(
    p_work_id TEXT,
    p_lease_token UUID,
    p_submission_id TEXT,
    p_submission_doc JSONB,
    p_precheck_status TEXT,
    p_precheck_doc JSONB,
    p_catalog_row JSONB,
    p_disabled_provision_row JSONB,
    p_smoke_attempt JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_provision RECORD;
    v_catalog public.research_lab_source_catalog%ROWTYPE;
    v_current RECORD;
    v_finish JSONB;
    v_disabled JSONB;
    v_completed_replay BOOLEAN := FALSE;
BEGIN
    IF p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR pg_catalog.jsonb_typeof(p_submission_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(p_precheck_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(p_catalog_row) <> 'object'
       OR pg_catalog.jsonb_typeof(p_disabled_provision_row) <> 'object'
       OR pg_catalog.jsonb_typeof(p_smoke_attempt) <> 'object'
       OR p_disabled_provision_row->>'provision_status' <> 'disabled'
       OR p_disabled_provision_row#>'{provision_doc,provider_registry_entry,active}'
          IS DISTINCT FROM 'false'::JSONB
       OR p_smoke_attempt->>'evaluation_mode' <> 'provisioning_smoke'
       OR p_smoke_attempt->>'result_status' <> 'passed' THEN
        RAISE EXCEPTION 'SOURCE_ADD current-provider rejection input is invalid';
    END IF;
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RETURN pg_catalog.jsonb_build_object('status', 'missing');
    END IF;
    IF v_work.work_status = 'completed' THEN
        IF v_work.work_kind <> 'provisioning_smoke'
           OR v_work.submission_id <> p_submission_id
           OR v_work.result_doc->>'status' <> 'not_eligible' THEN
            RAISE EXCEPTION 'SOURCE_ADD current-provider rejection terminal state differs';
        END IF;
        v_completed_replay := TRUE;
    ELSE
        IF v_work.work_status <> 'leased'
           OR v_work.work_kind <> 'provisioning_smoke'
           OR v_work.lease_token IS DISTINCT FROM p_lease_token THEN
            RETURN pg_catalog.jsonb_build_object('status', 'lease_lost');
        END IF;
        IF v_work.submission_id <> p_submission_id
           OR p_smoke_attempt->>'work_id' <> p_work_id
           OR COALESCE((p_smoke_attempt->>'attempt_number')::INTEGER, 0)
              <> v_work.attempt_count
           OR p_smoke_attempt->>'attempt_ref' <>
              'source_add_probe_attempt:' || pg_catalog.substr(
                  public.research_lab_source_add_jsonb_hash_v2(
                      pg_catalog.jsonb_build_object(
                          'prefix', 'source_add_probe_attempt',
                          'parts', pg_catalog.jsonb_build_array(
                              p_submission_id, p_work_id, v_work.attempt_count
                          )
                      )
                  ),
                  8,
                  16
              ) THEN
            RAISE EXCEPTION 'SOURCE_ADD current-provider smoke lease binding differs';
        END IF;

        v_finish := public.research_lab_source_add_finish_work(
            p_work_id,
            p_lease_token,
            'complete',
            'functional_probe_failed',
            p_submission_doc,
            p_precheck_status,
            p_precheck_doc,
            pg_catalog.jsonb_build_object('status', 'not_eligible'),
            p_smoke_attempt,
            '{}'::JSONB,
            '{}'::JSONB,
            '{}'::JSONB,
            NULL,
            FALSE
        );
        IF COALESCE(v_finish->>'status', '') <> 'completed' THEN
            RAISE EXCEPTION 'SOURCE_ADD current-provider work completion failed: %',
                COALESCE(v_finish->>'status', 'missing');
        END IF;
    END IF;

    SELECT * INTO v_smoke
    FROM public.research_lab_source_add_functional_probe_attempts
    WHERE attempt_ref = p_smoke_attempt->>'attempt_ref';
    IF NOT FOUND
       OR v_smoke.submission_id <> p_submission_id
       OR v_smoke.adapter_id <> v_work.adapter_id
       OR v_smoke.work_id <> p_work_id
       OR v_smoke.attempt_number <> v_work.attempt_count
       OR v_smoke.evaluation_mode <> 'provisioning_smoke'
       OR v_smoke.config_ref <> p_smoke_attempt->>'config_ref'
       OR v_smoke.result_status <> 'passed'
       OR v_smoke.route_hash <> p_smoke_attempt->>'route_hash'
       OR v_smoke.response_hash <> COALESCE(p_smoke_attempt->>'response_hash', '')
       OR v_smoke.status_class <> COALESCE(p_smoke_attempt->>'status_class', '')
       OR v_smoke.content_type <> COALESCE(p_smoke_attempt->>'content_type', '')
       OR v_smoke.byte_count <> COALESCE((p_smoke_attempt->>'byte_count')::INTEGER, 0)
       OR v_smoke.duration_ms <> COALESCE((p_smoke_attempt->>'duration_ms')::INTEGER, 0)
       OR v_smoke.retry_after_seconds <>
          COALESCE((p_smoke_attempt->>'retry_after_seconds')::INTEGER, 0)
       OR v_smoke.reason_codes <>
          COALESCE(p_smoke_attempt->'reason_codes', '[]'::JSONB)
       OR v_smoke.receipt_hash <> p_smoke_attempt->>'receipt_hash'
       OR v_smoke.business_artifact_hash <>
          p_smoke_attempt->>'business_artifact_hash'
       OR v_smoke.result_doc <> p_smoke_attempt->'result_doc' THEN
        RAISE EXCEPTION 'SOURCE_ADD current-provider persisted smoke differs';
    END IF;

    IF NOT v_completed_replay THEN
        v_disabled := public.research_lab_source_add_finalize_provision(
            p_submission_id,
            p_catalog_row,
            p_disabled_provision_row,
            '{}'::JSONB
        );
        IF COALESCE(v_disabled->>'status', '') NOT IN (
            'provisioned', 'already_provisioned'
        ) THEN
            RAISE EXCEPTION 'SOURCE_ADD current-provider disable failed: %',
                COALESCE(v_disabled->>'status', 'missing');
        END IF;
    END IF;

    SELECT * INTO v_catalog
    FROM public.research_lab_source_catalog
    WHERE adapter_id = v_work.adapter_id;
    SELECT * INTO v_provision
    FROM public.research_lab_source_add_provisioning_current
    WHERE submission_id = p_submission_id;
    SELECT * INTO v_current
    FROM public.research_lab_source_add_submission_current
    WHERE submission_id = p_submission_id;
    IF v_catalog.catalog_id IS NULL
       OR v_provision.provision_ref IS NULL
       OR v_current.submission_id IS NULL
       OR v_catalog.catalog_id <> p_catalog_row->>'catalog_id'
       OR v_catalog.miner_ref <> p_catalog_row->>'miner_ref'
       OR v_catalog.source_name <> p_catalog_row->>'source_name'
       OR v_catalog.source_kind <> p_catalog_row->>'source_kind'
       OR pg_catalog.to_jsonb(v_catalog.declared_base_domains) <>
          p_catalog_row->'declared_base_domains'
       OR v_catalog.registry_provider_id <>
          p_catalog_row->>'registry_provider_id'
       OR v_catalog.catalog_doc <> p_catalog_row->'catalog_doc'
       OR v_catalog.source_identity_hash <>
          p_catalog_row->>'source_identity_hash'
       OR v_provision.provision_ref <>
          p_disabled_provision_row->>'provision_ref'
       OR v_provision.catalog_id <> p_catalog_row->>'catalog_id'
       OR v_provision.submission_id <> p_submission_id
       OR v_provision.adapter_id <> v_work.adapter_id
       OR v_provision.miner_hotkey <>
          p_disabled_provision_row->>'miner_hotkey'
       OR v_provision.source_identity_hash <>
          p_disabled_provision_row->>'source_identity_hash'
       OR v_provision.registry_provider_id <>
          p_disabled_provision_row->>'registry_provider_id'
       OR v_provision.provision_status <> 'disabled'
       OR v_provision.provision_doc <>
          p_disabled_provision_row->'provision_doc'
       OR v_provision.credential_envelope <>
          COALESCE(
              p_disabled_provision_row->'credential_envelope', '{}'::JSONB
          )
       OR v_current.stage <> 'functional_probe_failed'
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_submissions terminal
           WHERE terminal.submission_id = p_submission_id
             AND terminal.stage IN ('accepted', 'leg1_queued', 'leg1_created')
       )
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_reward_intents intent
           WHERE intent.submission_id = p_submission_id
              OR intent.adapter_id = v_work.adapter_id
       )
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_reward_obligations reward
           WHERE reward.adapter_id = v_work.adapter_id
       )
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_work_items reward_work
           WHERE reward_work.submission_id = p_submission_id
             AND reward_work.work_kind = 'leg1_reward'
       ) THEN
        RAISE EXCEPTION 'SOURCE_ADD current-provider rejection idempotency differs';
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'not_eligible');
END;
$$;

-- The database owns the release economics. Caller-supplied cap values are
-- retained only for wire compatibility with N-1 and are never authoritative.
CREATE OR REPLACE FUNCTION public.research_lab_source_add_reserve_leg1_slot_v2(
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
            10,
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
        10,
        p_slot_lease_seconds
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_leg1_v2(
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
       OR COALESCE((p_reward->>'alpha_percent')::NUMERIC, 0) <> 1.0
       OR COALESCE((p_reward->>'reward_epochs')::INTEGER, 0) <> 20 THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 release economics differs';
    END IF;
    v_result := public.research_lab_source_add_finalize_leg1(
        p_intent_id,
        p_work_id,
        p_work_lease_token,
        p_slot_lease_token,
        10,
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

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_leg1_initial_event_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_leg INTEGER;
    v_current_seq INTEGER;
BEGIN
    SELECT reward.leg INTO v_leg
    FROM public.research_lab_source_add_reward_obligations reward
    WHERE reward.reward_ref = NEW.reward_ref;
    IF v_leg = 1 THEN
        SELECT prior.seq
        INTO v_current_seq
        FROM public.research_lab_source_add_reward_events prior
        WHERE prior.reward_ref = NEW.reward_ref
        ORDER BY prior.seq DESC, prior.created_at DESC
        LIMIT 1;
        IF v_current_seq IS NULL THEN
            IF NEW.seq <> 0
               OR NEW.reward_status <> 'active'
               OR NEW.reason <> 'leg1_functional_probe_passed' THEN
                RAISE EXCEPTION 'SOURCE_ADD Leg 1 initial reward event differs';
            END IF;
        ELSE
            IF EXISTS (
                SELECT 1
                FROM public.research_lab_source_add_reward_events terminal
                WHERE terminal.reward_ref = NEW.reward_ref
                  AND terminal.reward_status = 'stopped_forward'
            ) THEN
                RAISE EXCEPTION 'SOURCE_ADD Leg 1 stopped reward is terminal';
            END IF;
            IF NEW.seq <> v_current_seq + 1 THEN
                RAISE EXCEPTION 'SOURCE_ADD Leg 1 reward event sequence differs';
            END IF;
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_source_add_leg1_initial_event_v2
    ON public.research_lab_source_add_reward_events;
CREATE TRIGGER trg_source_add_leg1_initial_event_v2
    BEFORE INSERT ON public.research_lab_source_add_reward_events
    FOR EACH ROW EXECUTE FUNCTION
        public.enforce_research_lab_source_add_leg1_initial_event_v2();

-- Remove the stale, caller-authoritative entry points only after their V2
-- replacements exist in the same migration transaction.
REVOKE ALL ON FUNCTION public.research_lab_source_add_configure_probe(
    TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision(
    TEXT, JSONB, JSONB, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_source_add_reserve_leg1_slot(
    TEXT, TEXT, UUID, INTEGER, INTEGER
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_leg1(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) FROM service_role;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision_smoke(
    TEXT, UUID, TEXT, JSONB, JSONB, JSONB
) FROM service_role;

REVOKE ALL ON FUNCTION public.research_lab_source_add_configure_probe_v2(
    TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision_v2(
    TEXT, JSONB, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_reject_current_builtin_v2(
    TEXT, UUID, TEXT, JSONB, TEXT, JSONB, JSONB, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_reserve_leg1_slot_v2(
    TEXT, TEXT, UUID, INTEGER, INTEGER
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_leg1_v2(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_leg1_initial_event_v2()
    FROM PUBLIC, anon, authenticated, service_role;

GRANT EXECUTE ON FUNCTION public.research_lab_source_add_configure_probe_v2(
    TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_provision_v2(
    TEXT, JSONB, JSONB, JSONB
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_reject_current_builtin_v2(
    TEXT, UUID, TEXT, JSONB, TEXT, JSONB, JSONB, JSONB, JSONB
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_reserve_leg1_slot_v2(
    TEXT, TEXT, UUID, INTEGER, INTEGER
) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_leg1_v2(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) TO service_role;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1()
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
        'schema_version', 'leadpoet.source_add_post_accept_leg1_contract.v1',
        'daily_cap', 10,
        'leg1_alpha_percent', 1.0,
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
            'reserve_leg1_slot_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_reserve_leg1_slot_v2(text,text,uuid,integer,integer)'
            ) IS NOT NULL,
            'finalize_leg1_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_leg1_v2(text,text,uuid,uuid,integer,jsonb,jsonb)'
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
            'v2_callable', v_service_role_exists AND
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
                    'public.research_lab_source_add_reserve_leg1_slot_v2(text,text,uuid,integer,integer)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_leg1_v2(text,text,uuid,uuid,integer,jsonb,jsonb)',
                    'EXECUTE'
                ) AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)',
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

REVOKE ALL ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_post_accept_leg1_contract_v1()
    TO service_role;

COMMENT ON TABLE public.research_lab_source_add_reward_obligations IS
    'Append-only SOURCE_ADD reward legs: each finally accepted source may create its own 1% Leg 1 obligation; enabled implementation riders are separate obligations. Active percentages sum deterministically up to the Research Lab cap.';

COMMIT;
