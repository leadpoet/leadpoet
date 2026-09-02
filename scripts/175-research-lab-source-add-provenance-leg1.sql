-- Automatically queue SOURCE_ADD Leg 1 from exact attested provenance.
--
-- A provenance_precheck_passed result is the Leg 1 approval boundary. Live
-- functional probing and catalog provisioning remain independent, later
-- operator workflows. Existing post-provision Leg 1 rows retain their prior
-- approval kind and continue to use the v2/v3 compatibility entry points.

BEGIN;

SET LOCAL lock_timeout = '5s';

-- The trigger swap and historical reconciliation must not race a SOURCE_ADD
-- worker. The canonical restart applies this migration while the independent
-- SOURCE_ADD control is paused.
DO $$
BEGIN
    LOCK TABLE public.research_lab_source_add_control
        IN ACCESS EXCLUSIVE MODE NOWAIT;
    IF NOT COALESCE((
        SELECT paused
        FROM public.research_lab_source_add_control
        WHERE singleton
    ), FALSE) THEN
        RAISE EXCEPTION
            'SOURCE_ADD must be paused before provenance Leg 1 migration';
    END IF;
    LOCK TABLE
        public.research_lab_source_add_work_items,
        public.research_lab_source_add_submissions,
        public.research_lab_source_add_reward_intents,
        public.research_lab_source_add_reward_slots,
        public.research_lab_source_add_reward_obligations,
        public.research_lab_source_add_reward_events,
        public.research_lab_attested_execution_receipts_v2,
        public.research_lab_attested_receipt_edges_v2,
        public.research_lab_attested_business_artifact_links_v2
        IN SHARE ROW EXCLUSIVE MODE NOWAIT;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_work_items
        WHERE work_status = 'leased'
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD work is leased during provenance Leg 1 migration';
    END IF;
END;
$$;

ALTER TABLE public.research_lab_source_add_reward_intents
    ADD COLUMN IF NOT EXISTS approval_kind TEXT NOT NULL
        DEFAULT 'post_accept_functional_probe';
ALTER TABLE public.research_lab_source_add_reward_intents
    ADD COLUMN IF NOT EXISTS provenance_receipt_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE public.research_lab_source_add_reward_intents
    ADD COLUMN IF NOT EXISTS provenance_artifact_hash TEXT NOT NULL DEFAULT '';

-- The v4 reward worker deliberately consumes only provenance-era intents.
-- Refuse the cutover while any actionable legacy intent still needs the v3
-- worker, rather than strand it behind a mixed-version dispatcher.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_intents
        WHERE approval_kind = 'post_accept_functional_probe'
          AND intent_status IN ('queued', 'leased', 'retry_wait')
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD actionable legacy Leg 1 intent must drain before provenance Leg 1 migration';
    END IF;
END;
$$;

ALTER TABLE public.research_lab_source_add_reward_intents
    DROP CONSTRAINT IF EXISTS
        research_lab_source_add_reward_intents_approval_kind_check;
ALTER TABLE public.research_lab_source_add_reward_intents
    ADD CONSTRAINT research_lab_source_add_reward_intents_approval_kind_check
    CHECK (approval_kind IN (
        'post_accept_functional_probe',
        'provenance_precheck_passed'
    )) NOT VALID;
ALTER TABLE public.research_lab_source_add_reward_intents
    VALIDATE CONSTRAINT
        research_lab_source_add_reward_intents_approval_kind_check;

ALTER TABLE public.research_lab_source_add_reward_intents
    DROP CONSTRAINT IF EXISTS
        research_lab_source_add_reward_intents_provenance_hashes_check;
ALTER TABLE public.research_lab_source_add_reward_intents
    ADD CONSTRAINT
        research_lab_source_add_reward_intents_provenance_hashes_check
    CHECK (
        (
            approval_kind = 'post_accept_functional_probe'
            AND provenance_receipt_hash = ''
            AND provenance_artifact_hash = ''
        ) OR (
            approval_kind = 'provenance_precheck_passed'
            AND provenance_receipt_hash ~ '^sha256:[0-9a-f]{64}$'
            AND provenance_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
            -- The legacy columns stay populated for wire compatibility but
            -- may never point at authority different from the explicit kind.
            AND functional_receipt_hash = provenance_receipt_hash
            AND business_artifact_hash = provenance_artifact_hash
        )
    ) NOT VALID;
ALTER TABLE public.research_lab_source_add_reward_intents
    VALIDATE CONSTRAINT
        research_lab_source_add_reward_intents_provenance_hashes_check;

-- The slot kind lets the legacy and provenance trigger guards coexist. Old
-- reserve RPCs keep their default and therefore remain on the legacy guard.
ALTER TABLE public.research_lab_source_add_reward_slots
    ADD COLUMN IF NOT EXISTS approval_kind TEXT NOT NULL
        DEFAULT 'post_accept_functional_probe';
ALTER TABLE public.research_lab_source_add_reward_slots
    DROP CONSTRAINT IF EXISTS
        research_lab_source_add_reward_slots_approval_kind_check;
ALTER TABLE public.research_lab_source_add_reward_slots
    ADD CONSTRAINT research_lab_source_add_reward_slots_approval_kind_check
    CHECK (approval_kind IN (
        'post_accept_functional_probe',
        'provenance_precheck_passed'
    )) NOT VALID;
ALTER TABLE public.research_lab_source_add_reward_slots
    VALIDATE CONSTRAINT
        research_lab_source_add_reward_slots_approval_kind_check;

-- This view is the single measured database projection used by enqueue,
-- reconciliation, trigger guards, finalization, and the V2 reward reader.
-- Earliest exact passed provenance wins deterministically per submission.
CREATE OR REPLACE VIEW
    public.research_lab_source_add_provenance_leg1_authority_v1
WITH (security_invoker = true) AS
SELECT DISTINCT ON (history.submission_id)
    history.submission_id,
    history.adapter_id,
    history.miner_hotkey,
    history.precheck_status,
    receipt.receipt_hash AS provenance_receipt_hash,
    receipt.output_root AS provenance_artifact_hash,
    history.created_at AS provenance_created_at
FROM public.research_lab_source_add_submissions history
JOIN public.research_lab_attested_execution_receipts_v2 receipt
  ON receipt.receipt_hash =
     history.submission_doc->>'provenance_receipt_hash'
JOIN public.research_lab_attested_business_artifact_links_v2 link
  ON link.receipt_hash = receipt.receipt_hash
 AND link.artifact_kind = 'source_add_provenance'
 AND link.artifact_ref = history.submission_id
 AND link.artifact_hash = receipt.output_root
WHERE history.precheck_status = 'provenance_precheck_passed'
  AND history.precheck_doc->>'precheck_status' =
      'provenance_precheck_passed'
  AND history.submission_doc->>'provenance_receipt_hash'
      ~ '^sha256:[0-9a-f]{64}$'
  AND receipt.role = 'gateway_coordinator'
  AND receipt.purpose = 'research_lab.source_add_provenance.v2'
  AND receipt.receipt_status = 'succeeded'
  AND receipt.output_root ~ '^sha256:[0-9a-f]{64}$'
  AND receipt.receipt_doc->'parent_receipt_hashes' = '[]'::JSONB
  AND NOT EXISTS (
      SELECT 1
      FROM public.research_lab_attested_receipt_edges_v2 edge
      WHERE edge.child_receipt_hash = receipt.receipt_hash
  )
ORDER BY history.submission_id, history.seq ASC, history.created_at ASC;

REVOKE ALL ON TABLE
    public.research_lab_source_add_provenance_leg1_authority_v1
    FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE
    public.research_lab_source_add_provenance_leg1_authority_v1
    TO service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_provenance_leg1_authority_matches_v1(
        p_submission_id TEXT,
        p_adapter_id TEXT,
        p_miner_hotkey TEXT,
        p_receipt_hash TEXT,
        p_artifact_hash TEXT
    )
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT COUNT(*) = 1
    FROM public.research_lab_source_add_provenance_leg1_authority_v1 authority
    WHERE authority.submission_id = p_submission_id
      AND authority.adapter_id = p_adapter_id
      AND authority.miner_hotkey = p_miner_hotkey
      AND authority.precheck_status = 'provenance_precheck_passed'
      AND authority.provenance_receipt_hash = p_receipt_hash
      AND authority.provenance_artifact_hash = p_artifact_hash
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_provenance_leg1_authority_matches_v1(
        TEXT, TEXT, TEXT, TEXT, TEXT
    ) FROM PUBLIC, anon, authenticated, service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_enqueue_leg1_after_provenance_v1(
        p_submission_id TEXT,
        p_intent_id TEXT,
        p_reward_work_id TEXT,
        p_provenance_receipt_hash TEXT,
        p_provenance_artifact_hash TEXT
    )
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_authority RECORD;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_expected_intent_id TEXT;
    v_expected_work_id TEXT;
    v_existing_reward_ref TEXT;
    v_job_doc JSONB;
BEGIN
    IF p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR p_intent_id !~ '^source_add_reward_intent:[0-9a-f]{16}$'
       OR p_reward_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_provenance_receipt_hash !~ '^sha256:[0-9a-f]{64}$'
       OR p_provenance_artifact_hash !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance Leg 1 input is invalid';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-submission:' || p_submission_id,
            0
        )
    );
    SELECT reward.reward_ref INTO v_existing_reward_ref
    FROM public.research_lab_source_add_reward_obligations reward
    JOIN public.research_lab_source_add_submission_current current
      ON current.submission_id = p_submission_id
     AND current.adapter_id = reward.adapter_id
    WHERE reward.leg = 1;
    IF FOUND THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'already_created',
            'reward_ref', v_existing_reward_ref
        );
    END IF;
    SELECT * INTO v_authority
    FROM public.research_lab_source_add_provenance_leg1_authority_v1
    WHERE submission_id = p_submission_id;
    IF NOT FOUND
       OR v_authority.provenance_receipt_hash <>
          p_provenance_receipt_hash
       OR v_authority.provenance_artifact_hash <>
          p_provenance_artifact_hash THEN
        RETURN pg_catalog.jsonb_build_object('status', 'not_eligible');
    END IF;

    v_expected_intent_id := 'source_add_reward_intent:' || pg_catalog.substr(
        public.research_lab_source_add_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'prefix', 'source_add_reward_intent',
                'parts', pg_catalog.jsonb_build_array(
                    p_submission_id, v_authority.adapter_id, 1
                )
            )
        ),
        8,
        16
    );
    v_expected_work_id := 'source_add_work:' || pg_catalog.substr(
        public.research_lab_source_add_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'prefix', 'source_add_work',
                'parts', pg_catalog.jsonb_build_array(
                    p_submission_id, 'leg1_reward', v_expected_intent_id
                )
            )
        ),
        8,
        16
    );
    IF p_intent_id <> v_expected_intent_id
       OR p_reward_work_id <> v_expected_work_id THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance Leg 1 identity differs';
    END IF;

    INSERT INTO public.research_lab_source_add_reward_intents (
        intent_id,
        submission_id,
        adapter_id,
        miner_hotkey,
        intent_status,
        functional_receipt_hash,
        business_artifact_hash,
        approval_kind,
        provenance_receipt_hash,
        provenance_artifact_hash,
        available_at,
        created_at,
        updated_at
    ) VALUES (
        p_intent_id,
        p_submission_id,
        v_authority.adapter_id,
        v_authority.miner_hotkey,
        'queued',
        p_provenance_receipt_hash,
        p_provenance_artifact_hash,
        'provenance_precheck_passed',
        p_provenance_receipt_hash,
        p_provenance_artifact_hash,
        LEAST(v_authority.provenance_created_at, NOW()),
        v_authority.provenance_created_at,
        NOW()
    ) ON CONFLICT (adapter_id, leg) DO NOTHING;

    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE adapter_id = v_authority.adapter_id
      AND leg = 1;
    IF FOUND
       AND v_intent.submission_id = p_submission_id
       AND v_intent.miner_hotkey = v_authority.miner_hotkey
       AND v_intent.approval_kind = 'post_accept_functional_probe' THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', CASE
                WHEN v_intent.intent_status = 'finalized'
                    THEN 'already_created'
                ELSE 'legacy_existing'
            END,
            'intent_id', v_intent.intent_id,
            'reward_ref', v_intent.reward_ref
        );
    END IF;
    IF NOT FOUND
       OR v_intent.intent_id <> p_intent_id
       OR v_intent.submission_id <> p_submission_id
       OR v_intent.miner_hotkey <> v_authority.miner_hotkey
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR v_intent.provenance_receipt_hash <>
          p_provenance_receipt_hash
       OR v_intent.provenance_artifact_hash <>
          p_provenance_artifact_hash THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance Leg 1 intent idempotency differs';
    END IF;

    v_job_doc := pg_catalog.jsonb_build_object(
        'intent_id', p_intent_id,
        'approval_kind', 'provenance_precheck_passed',
        'provenance_receipt_hash', p_provenance_receipt_hash,
        'provenance_artifact_hash', p_provenance_artifact_hash
    );
    INSERT INTO public.research_lab_source_add_work_items (
        work_id,
        submission_id,
        adapter_id,
        work_kind,
        work_status,
        priority,
        available_at,
        job_doc,
        created_at,
        updated_at
    ) VALUES (
        p_reward_work_id,
        p_submission_id,
        v_authority.adapter_id,
        'leg1_reward',
        'queued',
        5,
        LEAST(v_authority.provenance_created_at, NOW()),
        v_job_doc,
        v_authority.provenance_created_at,
        NOW()
    ) ON CONFLICT (work_id) DO NOTHING;

    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_reward_work_id;
    IF NOT FOUND
       OR v_work.submission_id <> p_submission_id
       OR v_work.adapter_id <> v_authority.adapter_id
       OR v_work.work_kind <> 'leg1_reward'
       OR v_work.priority <> 5
       OR v_work.job_doc <> v_job_doc THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance Leg 1 work idempotency differs';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'status', CASE
            WHEN v_work.work_status = 'completed' THEN 'already_created'
            ELSE 'queued'
        END,
        'intent_id', p_intent_id,
        'work_id', p_reward_work_id,
        'work_status', v_work.work_status
    );
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_enqueue_leg1_after_provenance_v1(
        TEXT, TEXT, TEXT, TEXT, TEXT
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_enqueue_leg1_after_provenance_v1(
        TEXT, TEXT, TEXT, TEXT, TEXT
    ) TO service_role;

-- Keep the pass row and its reward work atomic for new submissions.  The
-- periodic reconciler below covers the already-committed N-1 transaction gap
-- and makes a lost HTTP response harmless without rerunning provenance.
CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_authority RECORD;
    v_intent_id TEXT;
    v_work_id TEXT;
    v_result JSONB;
BEGIN
    SELECT * INTO v_authority
    FROM public.research_lab_source_add_provenance_leg1_authority_v1
    WHERE submission_id = NEW.submission_id;
    IF NOT FOUND THEN
        IF EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_reward_obligations paid
            WHERE paid.adapter_id = NEW.adapter_id
              AND paid.leg = 1
        ) THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION
            'SOURCE_ADD provenance pass lacks exact attested authority';
    END IF;
    v_intent_id := 'source_add_reward_intent:' || pg_catalog.substr(
        public.research_lab_source_add_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'prefix', 'source_add_reward_intent',
                'parts', pg_catalog.jsonb_build_array(
                    NEW.submission_id,
                    v_authority.adapter_id,
                    1
                )
            )
        ),
        8,
        16
    );
    v_work_id := 'source_add_work:' || pg_catalog.substr(
        public.research_lab_source_add_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'prefix', 'source_add_work',
                'parts', pg_catalog.jsonb_build_array(
                    NEW.submission_id,
                    'leg1_reward',
                    v_intent_id
                )
            )
        ),
        8,
        16
    );
    v_result :=
        public.research_lab_source_add_enqueue_leg1_after_provenance_v1(
            NEW.submission_id,
            v_intent_id,
            v_work_id,
            v_authority.provenance_receipt_hash,
            v_authority.provenance_artifact_hash
        );
    IF COALESCE(v_result->>'status', '') NOT IN (
        'queued',
        'already_created',
        'legacy_existing'
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD automatic provenance Leg 1 enqueue failed: %',
            COALESCE(v_result->>'status', 'missing');
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_enqueue_provenance_leg1_v1
    ON public.research_lab_source_add_submissions;
CREATE TRIGGER trg_source_add_enqueue_provenance_leg1_v1
    AFTER INSERT ON public.research_lab_source_add_submissions
    FOR EACH ROW
    WHEN (
        NEW.precheck_status = 'provenance_precheck_passed'
        AND NEW.submission_doc->>'provenance_receipt_hash'
            ~ '^sha256:[0-9a-f]{64}$'
    )
    EXECUTE FUNCTION
        public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1();

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_reconcile_provenance_leg1_v1()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_authority RECORD;
    v_intent_id TEXT;
    v_work_id TEXT;
    v_result JSONB;
    v_eligible INTEGER := 0;
    v_queued INTEGER := 0;
    v_existing INTEGER := 0;
BEGIN
    FOR v_authority IN
        SELECT authority.*
        FROM public.research_lab_source_add_provenance_leg1_authority_v1
            authority
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_reward_obligations paid
            WHERE paid.adapter_id = authority.adapter_id
              AND paid.leg = 1
        )
        ORDER BY
            authority.provenance_created_at ASC,
            authority.submission_id ASC
    LOOP
        v_eligible := v_eligible + 1;
        v_intent_id := 'source_add_reward_intent:' || pg_catalog.substr(
            public.research_lab_source_add_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'prefix', 'source_add_reward_intent',
                    'parts', pg_catalog.jsonb_build_array(
                        v_authority.submission_id,
                        v_authority.adapter_id,
                        1
                    )
                )
            ),
            8,
            16
        );
        v_work_id := 'source_add_work:' || pg_catalog.substr(
            public.research_lab_source_add_jsonb_hash_v2(
                pg_catalog.jsonb_build_object(
                    'prefix', 'source_add_work',
                    'parts', pg_catalog.jsonb_build_array(
                        v_authority.submission_id,
                        'leg1_reward',
                        v_intent_id
                    )
                )
            ),
            8,
            16
        );
        v_result :=
            public.research_lab_source_add_enqueue_leg1_after_provenance_v1(
                v_authority.submission_id,
                v_intent_id,
                v_work_id,
                v_authority.provenance_receipt_hash,
                v_authority.provenance_artifact_hash
            );
        IF COALESCE(v_result->>'status', '') = 'queued' THEN
            v_queued := v_queued + 1;
        ELSIF COALESCE(v_result->>'status', '') IN (
            'already_created',
            'legacy_existing'
        ) THEN
            v_existing := v_existing + 1;
        ELSE
            RAISE EXCEPTION
                'SOURCE_ADD provenance Leg 1 reconcile failed: %',
                COALESCE(v_result->>'status', 'missing');
        END IF;
    END LOOP;
    RETURN pg_catalog.jsonb_build_object(
        'status', 'reconciled',
        'eligible_count', v_eligible,
        'queued_count', v_queued,
        'existing_count', v_existing
    );
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_reconcile_provenance_leg1_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_reconcile_provenance_leg1_v1()
    TO service_role;

-- Route only new provenance-authorized rows around the legacy post-accept
-- guards. The v2 triggers remain enabled for old rows and rollback traffic.
CREATE OR REPLACE FUNCTION
    public.enforce_research_lab_source_add_leg1_work_v3()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
BEGIN
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE intent_id = NEW.job_doc->>'intent_id';
    IF NOT FOUND
       OR NEW.work_kind <> 'leg1_reward'
       OR NEW.submission_id <> v_intent.submission_id
       OR NEW.adapter_id <> v_intent.adapter_id
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR NEW.job_doc <> pg_catalog.jsonb_build_object(
           'intent_id', v_intent.intent_id,
           'approval_kind', 'provenance_precheck_passed',
           'provenance_receipt_hash', v_intent.provenance_receipt_hash,
           'provenance_artifact_hash', v_intent.provenance_artifact_hash
       )
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD Leg 1 work requires exact provenance approval';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_leg1_work_v3()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_work_v2
    ON public.research_lab_source_add_work_items;
CREATE TRIGGER trg_source_add_leg1_work_v2
    BEFORE INSERT ON public.research_lab_source_add_work_items
    FOR EACH ROW
    WHEN (
        NEW.work_kind = 'leg1_reward'
        AND COALESCE(
            NEW.job_doc->>'approval_kind',
            'post_accept_functional_probe'
        ) <> 'provenance_precheck_passed'
    )
    EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_work_v2();

DROP TRIGGER IF EXISTS trg_source_add_leg1_work_v3
    ON public.research_lab_source_add_work_items;
CREATE TRIGGER trg_source_add_leg1_work_v3
    BEFORE INSERT ON public.research_lab_source_add_work_items
    FOR EACH ROW
    WHEN (
        NEW.work_kind = 'leg1_reward'
        AND NEW.job_doc->>'approval_kind' = 'provenance_precheck_passed'
    )
    EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_work_v3();

CREATE OR REPLACE FUNCTION
    public.enforce_research_lab_source_add_leg1_slot_v3()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
BEGIN
    IF NEW.slot_status <> 'reserved' THEN
        RETURN NEW;
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE intent_id = NEW.intent_id;
    IF NOT FOUND
       OR NEW.approval_kind <> 'provenance_precheck_passed'
       OR v_intent.approval_kind <> NEW.approval_kind
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD Leg 1 slot requires exact provenance approval';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_leg1_slot_v3()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_slot_v2
    ON public.research_lab_source_add_reward_slots;
CREATE TRIGGER trg_source_add_leg1_slot_v2
    BEFORE INSERT OR UPDATE OF slot_status, intent_id
    ON public.research_lab_source_add_reward_slots
    FOR EACH ROW
    WHEN (NEW.approval_kind <> 'provenance_precheck_passed')
    EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_slot_v2();

DROP TRIGGER IF EXISTS trg_source_add_leg1_slot_v3
    ON public.research_lab_source_add_reward_slots;
CREATE TRIGGER trg_source_add_leg1_slot_v3
    BEFORE INSERT OR UPDATE OF slot_status, intent_id
    ON public.research_lab_source_add_reward_slots
    FOR EACH ROW
    WHEN (NEW.approval_kind = 'provenance_precheck_passed')
    EXECUTE FUNCTION public.enforce_research_lab_source_add_leg1_slot_v3();

CREATE OR REPLACE FUNCTION
    public.enforce_research_lab_source_add_leg1_obligation_v3()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_expected_trigger JSONB;
    v_expected_projection JSONB;
    v_expected_decision_hash TEXT;
    v_expected_reward_ref TEXT;
    v_alpha_json JSONB;
    v_decision RECORD;
    v_decision_count INTEGER;
BEGIN
    IF NEW.leg <> 1
       OR COALESCE(
           (NEW.trigger_evidence_doc->>'provenance_precheck_passed')::BOOLEAN,
           FALSE
       ) IS NOT TRUE THEN
        RETURN NEW;
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE adapter_id = NEW.adapter_id
      AND leg = 1;
    IF NOT FOUND
       OR v_intent.miner_hotkey <> NEW.miner_hotkey
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 provenance owner differs';
    END IF;
    IF NEW.catalog_id IS NOT NULL
       OR NEW.reward_kind <> 'source_acceptance'
       OR NEW.alpha_percent <> 0.2
       OR NEW.reward_epochs <> 20
       OR NEW.start_epoch < 0
       OR NEW.public_label <> 'Source acceptance reward' THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 provenance economics differ';
    END IF;
    v_expected_reward_ref := 'source_add_reward:' || pg_catalog.substr(
        public.research_lab_source_add_jsonb_hash_v2(
            pg_catalog.jsonb_build_object(
                'adapter_id', NEW.adapter_id,
                'leg', 1
            )
        ),
        8,
        16
    );
    IF NEW.reward_ref <> v_expected_reward_ref THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 provenance reward identity differs';
    END IF;
    v_expected_trigger := pg_catalog.jsonb_build_object(
        'provenance_precheck_passed', TRUE,
        'submission_id', v_intent.submission_id,
        'precheck_status', 'provenance_precheck_passed',
        'provenance_receipt_hash', v_intent.provenance_receipt_hash,
        'provenance_artifact_hash', v_intent.provenance_artifact_hash,
        'provenance_result_hash', v_intent.provenance_artifact_hash
    );
    IF NEW.trigger_evidence_doc IS DISTINCT FROM v_expected_trigger THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 provenance evidence differs';
    END IF;

    v_alpha_json := pg_catalog.to_jsonb(
        NEW.alpha_percent::DOUBLE PRECISION
    );
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
    SELECT COUNT(*)::INTEGER INTO v_decision_count
    FROM public.research_lab_attested_business_artifact_links_v2 link
    WHERE link.artifact_kind = 'source_add_reward_decision'
      AND link.artifact_ref = NEW.reward_ref
      AND link.artifact_hash = v_expected_decision_hash;
    IF v_decision_count <> 1 THEN
        RAISE EXCEPTION
            'SOURCE_ADD Leg 1 exact provenance decision is missing or ambiguous';
    END IF;
    SELECT
        receipt.receipt_hash,
        receipt.output_root,
        receipt.receipt_doc,
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
          IS DISTINCT FROM pg_catalog.jsonb_build_array(
              v_intent.provenance_receipt_hash
          )
       OR (
           SELECT COUNT(*)
           FROM public.research_lab_attested_receipt_edges_v2 edge
           WHERE edge.child_receipt_hash = v_decision.receipt_hash
       ) <> 1
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_receipt_edges_v2 edge
           WHERE edge.child_receipt_hash = v_decision.receipt_hash
             AND edge.parent_receipt_hash =
                 v_intent.provenance_receipt_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD Leg 1 provenance decision ancestry differs';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_leg1_obligation_v3()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_obligation_v2
    ON public.research_lab_source_add_reward_obligations;
CREATE TRIGGER trg_source_add_leg1_obligation_v2
    BEFORE INSERT ON public.research_lab_source_add_reward_obligations
    FOR EACH ROW
    WHEN (
        COALESCE(
            (NEW.trigger_evidence_doc->>'provenance_precheck_passed')::BOOLEAN,
            FALSE
        ) IS NOT TRUE
    )
    EXECUTE FUNCTION
        public.enforce_research_lab_source_add_leg1_obligation_v2();

DROP TRIGGER IF EXISTS trg_source_add_leg1_obligation_v3
    ON public.research_lab_source_add_reward_obligations;
CREATE TRIGGER trg_source_add_leg1_obligation_v3
    BEFORE INSERT ON public.research_lab_source_add_reward_obligations
    FOR EACH ROW
    WHEN (
        COALESCE(
            (NEW.trigger_evidence_doc->>'provenance_precheck_passed')::BOOLEAN,
            FALSE
        ) IS TRUE
    )
    EXECUTE FUNCTION
        public.enforce_research_lab_source_add_leg1_obligation_v3();

CREATE OR REPLACE FUNCTION
    public.enforce_research_lab_source_add_leg1_initial_event_v3()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_reward public.research_lab_source_add_reward_obligations%ROWTYPE;
BEGIN
    SELECT * INTO v_reward
    FROM public.research_lab_source_add_reward_obligations
    WHERE reward_ref = NEW.reward_ref;
    IF NOT FOUND
       OR v_reward.leg <> 1
       OR COALESCE(
           (
               v_reward.trigger_evidence_doc
               ->>'provenance_precheck_passed'
           )::BOOLEAN,
           FALSE
       ) IS NOT TRUE
       OR NEW.seq <> 0
       OR NEW.reward_status <> 'active'
       OR NEW.reason <> 'leg1_provenance_precheck_passed'
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_reward_events prior
           WHERE prior.reward_ref = NEW.reward_ref
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD Leg 1 provenance initial event differs';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_leg1_initial_event_v3()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_leg1_initial_event_v2
    ON public.research_lab_source_add_reward_events;
CREATE TRIGGER trg_source_add_leg1_initial_event_v2
    BEFORE INSERT ON public.research_lab_source_add_reward_events
    FOR EACH ROW
    WHEN (NEW.reason <> 'leg1_provenance_precheck_passed')
    EXECUTE FUNCTION
        public.enforce_research_lab_source_add_leg1_initial_event_v2();

DROP TRIGGER IF EXISTS trg_source_add_leg1_initial_event_v3
    ON public.research_lab_source_add_reward_events;
CREATE TRIGGER trg_source_add_leg1_initial_event_v3
    BEFORE INSERT ON public.research_lab_source_add_reward_events
    FOR EACH ROW
    WHEN (NEW.reason = 'leg1_provenance_precheck_passed')
    EXECUTE FUNCTION
        public.enforce_research_lab_source_add_leg1_initial_event_v3();

-- Provenance approval does not freeze the probe configuration. This separate
-- candidate RPC leaves the exact legacy v2 behavior available for rollback.
CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_configure_probe_v3(
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
            'source-add-submission:' || p_submission_id,
            0
        )
    );
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submissions history
        WHERE history.submission_id = p_submission_id
          AND history.stage IN (
              'accepted',
              'rejected',
              'rejected_precheck',
              'functional_probe_failed'
          )
    ) THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'final_approval_frozen'
        );
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

REVOKE ALL ON FUNCTION public.research_lab_source_add_configure_probe_v3(
    TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_configure_probe_v3(
    TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT
) TO service_role;

-- Provisioning remains a separate operator decision after the automatic Leg 1
-- reward.  Mark only a provenance-era eligible row inside the trusted wrapper
-- so the legacy v2 eligibility guard remains available unchanged for rollback.
CREATE OR REPLACE FUNCTION
    public.enforce_research_lab_source_add_eligible_v3()
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
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR NEW.provision_doc->>'leg1_approval_kind' <>
          'provenance_precheck_passed'
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       )
       OR v_functional.attempt_ref IS NULL
       OR v_smoke.attempt_ref IS NULL
       OR v_functional.adapter_id <> NEW.adapter_id
       OR v_functional.result_status <> 'passed'
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
             AND receipt.purpose =
                 'research_lab.source_add_functional_probe.v2'
             AND receipt.receipt_status = 'succeeded'
             AND receipt.output_root =
                 v_functional.business_artifact_hash
             AND link.artifact_kind = 'source_add_functional_probe'
             AND link.artifact_ref = v_functional.attempt_ref
             AND link.artifact_hash =
                 v_functional.business_artifact_hash
       )
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_execution_receipts_v2 receipt
           JOIN public.research_lab_attested_business_artifact_links_v2 link
             ON link.receipt_hash = receipt.receipt_hash
           WHERE receipt.receipt_hash = v_smoke.receipt_hash
             AND receipt.role = 'gateway_coordinator'
             AND receipt.purpose =
                 'research_lab.source_add_functional_probe.v2'
             AND receipt.receipt_status = 'succeeded'
             AND receipt.output_root = v_smoke.business_artifact_hash
             AND link.artifact_kind = 'source_add_provisioning_smoke'
             AND link.artifact_ref = v_smoke.attempt_ref
             AND link.artifact_hash = v_smoke.business_artifact_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD eligible provisioning lacks exact probe authority';
    END IF;
    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION
    public.enforce_research_lab_source_add_eligible_v3()
    FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_source_add_eligible_v2
    ON public.research_lab_source_add_provisioning_events;
CREATE TRIGGER trg_source_add_eligible_v2
    BEFORE INSERT ON public.research_lab_source_add_provisioning_events
    FOR EACH ROW
    WHEN (
        COALESCE(
            NEW.provision_doc->>'leg1_approval_kind',
            'post_accept_functional_probe'
        ) <> 'provenance_precheck_passed'
    )
    EXECUTE FUNCTION public.enforce_research_lab_source_add_eligible_v2();

DROP TRIGGER IF EXISTS trg_source_add_eligible_v3
    ON public.research_lab_source_add_provisioning_events;
CREATE TRIGGER trg_source_add_eligible_v3
    BEFORE INSERT ON public.research_lab_source_add_provisioning_events
    FOR EACH ROW
    WHEN (
        NEW.provision_doc->>'leg1_approval_kind' =
        'provenance_precheck_passed'
    )
    EXECUTE FUNCTION public.enforce_research_lab_source_add_eligible_v3();

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_enqueue_provision_smoke_v2(
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
    v_result JSONB;
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_terminal_result_status TEXT;
    v_terminal_status TEXT;
BEGIN
    -- The original function remains the exact validator and happy-path
    -- implementation.  Only its legacy final-approval freeze is relaxed for
    -- a conclusively failed smoke owned by an exact provenance-era intent.
    v_result := public.research_lab_source_add_enqueue_provision_smoke(
        p_work_id,
        p_submission_id,
        p_config_ref,
        p_host_hash,
        p_catalog_row,
        p_provision_row
    );
    IF COALESCE(v_result->>'status', '') <>
       'terminal_retry_not_allowed' THEN
        RETURN v_result;
    END IF;
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE submission_id = p_submission_id
      AND adapter_id = v_work.adapter_id
      AND leg = 1;
    v_terminal_result_status := COALESCE(
        v_work.result_doc->>'result_status',
        ''
    );
    v_terminal_status := COALESCE(v_work.result_doc->>'status', '');
    IF NOT FOUND
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       )
       OR v_work.work_status <> 'completed'
       OR v_work.work_kind <> 'provisioning_smoke'
       OR v_work.submission_id <> p_submission_id
       OR v_work.attempt_count >= 20
       OR NOT (
           (
               v_terminal_result_status IN (
                   'failed',
                   'manual_review',
                   'awaiting_operator',
                   'retryable'
               )
               AND EXISTS (
                   SELECT 1
                   FROM public.research_lab_source_add_functional_probe_attempts
                       attempt
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
                   FROM public.research_lab_source_add_functional_probe_attempts
                       attempt
                   WHERE attempt.work_id = p_work_id
                     AND attempt.attempt_number = v_work.attempt_count
                     AND attempt.evaluation_mode = 'provisioning_smoke'
                     AND attempt.result_status = 'passed'
               )
           )
       )
       OR EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_provisioning_events eligible
           WHERE eligible.adapter_id = v_work.adapter_id
             AND eligible.provision_status =
                 'provisioned_autoresearch_eligible'
       ) THEN
        RETURN v_result;
    END IF;
    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'queued',
        available_at = NOW(),
        lease_token = NULL,
        leased_by = '',
        lease_expires_at = NULL,
        completed_at = NULL,
        result_doc = pg_catalog.jsonb_build_object(
            'status', 'operator_requeued',
            'prior_result_status', COALESCE(
                NULLIF(v_terminal_result_status, ''),
                v_terminal_status
            )
        ),
        updated_at = NOW()
    WHERE work_id = p_work_id;
    RETURN pg_catalog.jsonb_build_object(
        'status', 'queued',
        'work_id', p_work_id,
        'work_status', 'queued',
        'requeued', TRUE
    );
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_enqueue_provision_smoke_v2(
        TEXT, TEXT, TEXT, TEXT, JSONB, JSONB
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_enqueue_provision_smoke_v2(
        TEXT, TEXT, TEXT, TEXT, JSONB, JSONB
    ) TO service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_finalize_provision_v3(
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
DECLARE
    v_provision_row JSONB := p_provision_row;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-submission:' || p_submission_id,
            0
        )
    );
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submissions history
        WHERE history.submission_id = p_submission_id
          AND history.stage IN (
              'accepted',
              'rejected',
              'rejected_precheck',
              'functional_probe_failed'
          )
    ) THEN
        RETURN pg_catalog.jsonb_build_object(
            'status', 'final_approval_frozen'
        );
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE submission_id = p_submission_id
      AND leg = 1;
    IF p_provision_row->>'provision_status' =
       'provisioned_autoresearch_eligible' THEN
        IF NOT FOUND
           OR v_intent.approval_kind <> 'provenance_precheck_passed'
           OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
               v_intent.submission_id,
               v_intent.adapter_id,
               v_intent.miner_hotkey,
               v_intent.provenance_receipt_hash,
               v_intent.provenance_artifact_hash
           ) THEN
            RAISE EXCEPTION
                'SOURCE_ADD provenance provisioning authority differs';
        END IF;
        v_provision_row := pg_catalog.jsonb_set(
            p_provision_row,
            '{provision_doc,leg1_approval_kind}',
            pg_catalog.to_jsonb('provenance_precheck_passed'::TEXT),
            TRUE
        );
    END IF;
    RETURN public.research_lab_source_add_finalize_provision(
        p_submission_id,
        p_catalog_row,
        v_provision_row,
        p_smoke_attempt
    );
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_finalize_provision_v3(
        TEXT, JSONB, JSONB, JSONB
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_finalize_provision_v3(
        TEXT, JSONB, JSONB, JSONB
    ) TO service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_finalize_provision_smoke_v3(
        p_work_id TEXT,
        p_lease_token UUID,
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
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_expected_job JSONB;
    v_provision_row JSONB;
    v_result JSONB;
BEGIN
    IF p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR pg_catalog.jsonb_typeof(p_catalog_row) <> 'object'
       OR pg_catalog.jsonb_typeof(p_provision_row) <> 'object'
       OR pg_catalog.jsonb_typeof(p_smoke_attempt) <> 'object' THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance provisioning smoke input is invalid';
    END IF;
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND
       OR v_work.work_status <> 'leased'
       OR v_work.work_kind <> 'provisioning_smoke'
       OR v_work.lease_token IS DISTINCT FROM p_lease_token THEN
        RETURN pg_catalog.jsonb_build_object('status', 'lease_lost');
    END IF;
    v_expected_job := pg_catalog.jsonb_build_object(
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
                          p_submission_id,
                          p_work_id,
                          v_work.attempt_count
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
        RAISE EXCEPTION
            'SOURCE_ADD provenance provisioning smoke lease binding differs';
    END IF;
    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE submission_id = p_submission_id
      AND adapter_id = v_work.adapter_id
      AND leg = 1;
    IF NOT FOUND
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance provisioning authority differs';
    END IF;
    v_provision_row := pg_catalog.jsonb_set(
        p_provision_row,
        '{provision_doc,leg1_approval_kind}',
        pg_catalog.to_jsonb('provenance_precheck_passed'::TEXT),
        TRUE
    );
    v_result := public.research_lab_source_add_finalize_provision(
        p_submission_id,
        p_catalog_row,
        v_provision_row,
        p_smoke_attempt
    );
    IF COALESCE(v_result->>'status', '') NOT IN (
        'provisioned',
        'already_provisioned'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance provisioning failed: %',
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
       OR v_smoke.response_hash <>
          COALESCE(p_smoke_attempt->>'response_hash', '')
       OR v_smoke.status_class <>
          COALESCE(p_smoke_attempt->>'status_class', '')
       OR v_smoke.content_type <>
          COALESCE(p_smoke_attempt->>'content_type', '')
       OR v_smoke.byte_count <>
          COALESCE((p_smoke_attempt->>'byte_count')::INTEGER, 0)
       OR v_smoke.duration_ms <>
          COALESCE((p_smoke_attempt->>'duration_ms')::INTEGER, 0)
       OR v_smoke.retry_after_seconds <>
          COALESCE((p_smoke_attempt->>'retry_after_seconds')::INTEGER, 0)
       OR v_smoke.reason_codes <>
          COALESCE(p_smoke_attempt->'reason_codes', '[]'::JSONB)
       OR v_smoke.receipt_hash <> p_smoke_attempt->>'receipt_hash'
       OR v_smoke.business_artifact_hash <>
          p_smoke_attempt->>'business_artifact_hash'
       OR v_smoke.result_doc <> p_smoke_attempt->'result_doc' THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance persisted smoke differs from lease';
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

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_finalize_provision_smoke_v3(
        TEXT, UUID, TEXT, JSONB, JSONB, JSONB
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_finalize_provision_smoke_v3(
        TEXT, UUID, TEXT, JSONB, JSONB, JSONB
    ) TO service_role;

-- A source can become part of the current model after its credible provenance
-- was attested and Leg 1 was queued or finalized.  That late catalog result
-- disables only provisioning: the provenance-era reward is append-only and
-- remains unchanged.  The legacy v2 rejection function remains available for
-- submissions which never crossed the provenance approval boundary.
CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_reject_current_builtin_v3(
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
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_reward public.research_lab_source_add_reward_obligations%ROWTYPE;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_provision RECORD;
    v_catalog public.research_lab_source_catalog%ROWTYPE;
    v_current RECORD;
    v_finish JSONB;
    v_disabled JSONB;
    v_rewards_before JSONB;
    v_rewards_after JSONB;
    v_reward_events_before JSONB;
    v_reward_events_after JSONB;
    v_completed_replay BOOLEAN := FALSE;
    v_reward_found BOOLEAN := FALSE;
BEGIN
    IF p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR pg_catalog.jsonb_typeof(p_submission_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(p_precheck_doc) <> 'object'
       OR pg_catalog.jsonb_typeof(p_catalog_row) <> 'object'
       OR pg_catalog.jsonb_typeof(p_disabled_provision_row) <> 'object'
       OR pg_catalog.jsonb_typeof(p_smoke_attempt) <> 'object'
       OR p_precheck_status IS DISTINCT FROM
          'provenance_precheck_passed'
       OR p_precheck_doc->>'precheck_status' IS DISTINCT FROM
          'provenance_precheck_passed'
       OR p_disabled_provision_row->>'provision_status' IS DISTINCT FROM
          'disabled'
       OR p_disabled_provision_row#>'{provision_doc,provider_registry_entry,active}'
          IS DISTINCT FROM 'false'::JSONB
       OR p_smoke_attempt->>'evaluation_mode' IS DISTINCT FROM
          'provisioning_smoke'
       OR p_smoke_attempt->>'result_status' IS DISTINCT FROM 'passed' THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance current-provider rejection input is invalid';
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
            RAISE EXCEPTION
                'SOURCE_ADD provenance current-provider terminal state differs';
        END IF;
        v_completed_replay := TRUE;
    ELSE
        IF v_work.work_status <> 'leased'
           OR v_work.work_kind <> 'provisioning_smoke'
           OR v_work.lease_token IS DISTINCT FROM p_lease_token THEN
            RETURN pg_catalog.jsonb_build_object('status', 'lease_lost');
        END IF;
    END IF;
    IF v_work.submission_id <> p_submission_id
       OR p_smoke_attempt->>'work_id' IS DISTINCT FROM p_work_id
       OR COALESCE((p_smoke_attempt->>'attempt_number')::INTEGER, 0)
          <> v_work.attempt_count
       OR p_smoke_attempt->>'attempt_ref' IS DISTINCT FROM
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
       OR v_work.job_doc->>'config_ref' IS DISTINCT FROM
          p_smoke_attempt->>'config_ref'
       OR v_work.job_doc->'catalog_row' IS DISTINCT FROM p_catalog_row
       OR v_work.job_doc#>'{provision_row,provision_status}'
          IS DISTINCT FROM
          pg_catalog.to_jsonb('provisioned_autoresearch_eligible'::TEXT)
       OR (
           p_disabled_provision_row
           - 'provision_ref'
           - 'provision_status'
           - 'provision_doc'
       ) IS DISTINCT FROM (
           (v_work.job_doc->'provision_row')
           - 'provision_ref'
           - 'provision_status'
           - 'provision_doc'
       )
       OR p_disabled_provision_row->'provision_doc' IS DISTINCT FROM
          pg_catalog.jsonb_set(
              v_work.job_doc#>'{provision_row,provision_doc}',
              '{provider_registry_entry,active}',
              'false'::JSONB,
              TRUE
          ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance current-provider smoke binding differs';
    END IF;

    SELECT * INTO v_intent
    FROM public.research_lab_source_add_reward_intents
    WHERE submission_id = p_submission_id
      AND adapter_id = v_work.adapter_id
      AND leg = 1
    FOR UPDATE;
    IF NOT FOUND
       OR v_intent.approval_kind <> 'provenance_precheck_passed'
       OR v_intent.intent_status NOT IN (
           'queued', 'leased', 'retry_wait', 'finalized'
       )
       OR p_submission_doc->>'provenance_receipt_hash' IS DISTINCT FROM
          v_intent.provenance_receipt_hash
       OR (
           COALESCE(
               p_submission_doc->>'provenance_artifact_hash', ''
           ) <> ''
           AND p_submission_doc->>'provenance_artifact_hash'
               IS DISTINCT FROM v_intent.provenance_artifact_hash
       )
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       )
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_work_items reward_work
           WHERE reward_work.submission_id = p_submission_id
             AND reward_work.adapter_id = v_work.adapter_id
             AND reward_work.work_kind = 'leg1_reward'
             AND reward_work.job_doc->>'intent_id' = v_intent.intent_id
             AND reward_work.job_doc->>'approval_kind' =
                 'provenance_precheck_passed'
             AND reward_work.job_doc->>'provenance_receipt_hash' =
                 v_intent.provenance_receipt_hash
             AND reward_work.job_doc->>'provenance_artifact_hash' =
                 v_intent.provenance_artifact_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance current-provider reward authority differs';
    END IF;

    SELECT * INTO v_reward
    FROM public.research_lab_source_add_reward_obligations
    WHERE adapter_id = v_work.adapter_id
      AND leg = 1;
    v_reward_found := FOUND;
    IF (v_intent.intent_status = 'finalized') IS DISTINCT FROM v_reward_found
       OR (
           v_reward_found
           AND (
               v_intent.reward_ref IS DISTINCT FROM v_reward.reward_ref
               OR v_reward.miner_hotkey <> v_intent.miner_hotkey
               OR v_reward.catalog_id IS NOT NULL
               OR v_reward.reward_kind <> 'source_acceptance'
               OR v_reward.alpha_percent <> 0.2
               OR v_reward.reward_epochs <> 20
               OR v_reward.trigger_evidence_doc <>
                  pg_catalog.jsonb_build_object(
                      'provenance_precheck_passed', TRUE,
                      'submission_id', v_intent.submission_id,
                      'precheck_status', 'provenance_precheck_passed',
                      'provenance_receipt_hash',
                          v_intent.provenance_receipt_hash,
                      'provenance_artifact_hash',
                          v_intent.provenance_artifact_hash,
                      'provenance_result_hash',
                          v_intent.provenance_artifact_hash
                  )
           )
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance current-provider reward state differs';
    END IF;

    SELECT COALESCE(
        pg_catalog.jsonb_agg(
            pg_catalog.to_jsonb(reward) ORDER BY reward.reward_ref
        ),
        '[]'::JSONB
    ) INTO v_rewards_before
    FROM public.research_lab_source_add_reward_obligations reward
    WHERE reward.adapter_id = v_work.adapter_id;
    SELECT COALESCE(
        pg_catalog.jsonb_agg(
            pg_catalog.to_jsonb(event)
            ORDER BY event.reward_ref, event.seq
        ),
        '[]'::JSONB
    ) INTO v_reward_events_before
    FROM public.research_lab_source_add_reward_events event
    JOIN public.research_lab_source_add_reward_obligations reward
      ON reward.reward_ref = event.reward_ref
    WHERE reward.adapter_id = v_work.adapter_id;

    IF NOT v_completed_replay THEN
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
            RAISE EXCEPTION
                'SOURCE_ADD provenance current-provider work completion failed: %',
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
       OR v_smoke.response_hash <>
          COALESCE(p_smoke_attempt->>'response_hash', '')
       OR v_smoke.status_class <>
          COALESCE(p_smoke_attempt->>'status_class', '')
       OR v_smoke.content_type <>
          COALESCE(p_smoke_attempt->>'content_type', '')
       OR v_smoke.byte_count <>
          COALESCE((p_smoke_attempt->>'byte_count')::INTEGER, 0)
       OR v_smoke.duration_ms <>
          COALESCE((p_smoke_attempt->>'duration_ms')::INTEGER, 0)
       OR v_smoke.retry_after_seconds <>
          COALESCE((p_smoke_attempt->>'retry_after_seconds')::INTEGER, 0)
       OR v_smoke.reason_codes <>
          COALESCE(p_smoke_attempt->'reason_codes', '[]'::JSONB)
       OR v_smoke.receipt_hash <> p_smoke_attempt->>'receipt_hash'
       OR v_smoke.business_artifact_hash <>
          p_smoke_attempt->>'business_artifact_hash'
       OR v_smoke.result_doc <> p_smoke_attempt->'result_doc' THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance current-provider persisted smoke differs';
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
            RAISE EXCEPTION
                'SOURCE_ADD provenance current-provider disable failed: %',
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
    SELECT COALESCE(
        pg_catalog.jsonb_agg(
            pg_catalog.to_jsonb(reward) ORDER BY reward.reward_ref
        ),
        '[]'::JSONB
    ) INTO v_rewards_after
    FROM public.research_lab_source_add_reward_obligations reward
    WHERE reward.adapter_id = v_work.adapter_id;
    SELECT COALESCE(
        pg_catalog.jsonb_agg(
            pg_catalog.to_jsonb(event)
            ORDER BY event.reward_ref, event.seq
        ),
        '[]'::JSONB
    ) INTO v_reward_events_after
    FROM public.research_lab_source_add_reward_events event
    JOIN public.research_lab_source_add_reward_obligations reward
      ON reward.reward_ref = event.reward_ref
    WHERE reward.adapter_id = v_work.adapter_id;
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
       OR (
           NOT v_completed_replay
           AND v_current.stage <> 'functional_probe_failed'
       )
       OR (
           v_completed_replay
           AND v_current.stage NOT IN (
               'functional_probe_failed', 'leg1_created'
           )
       )
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_source_add_submissions rejected
           WHERE rejected.submission_id = p_submission_id
             AND rejected.stage = 'functional_probe_failed'
       )
       OR v_rewards_after IS DISTINCT FROM v_rewards_before
       OR v_reward_events_after IS DISTINCT FROM v_reward_events_before
       THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance current-provider rejection idempotency differs';
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'not_eligible');
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_reject_current_builtin_v3(
        TEXT, UUID, TEXT, JSONB, TEXT, JSONB, JSONB, JSONB, JSONB
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_reject_current_builtin_v3(
        TEXT, UUID, TEXT, JSONB, TEXT, JSONB, JSONB, JSONB, JSONB
    ) TO service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_reserve_leg1_slot_v4(
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
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_existing_slot public.research_lab_source_add_reward_slots%ROWTYPE;
    v_existing_reward_ref TEXT;
    v_oldest_work_id TEXT;
    v_day DATE := (NOW() AT TIME ZONE 'UTC')::DATE;
    v_created INTEGER;
    v_reserved INTEGER;
    v_slot INTEGER;
    v_token UUID := gen_random_uuid();
    v_retry_at TIMESTAMPTZ := NOW() + INTERVAL '5 seconds';
BEGIN
    -- p_daily_cap is intentionally non-authoritative wire compatibility.
    IF p_slot_lease_seconds < 30 OR p_slot_lease_seconds > 1800 THEN
        RAISE EXCEPTION 'SOURCE_ADD reward slot policy is invalid';
    END IF;
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND
       OR v_work.work_status <> 'leased'
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
    IF v_intent.approval_kind <> 'provenance_precheck_passed'
       OR v_work.submission_id <> v_intent.submission_id
       OR v_work.adapter_id <> v_intent.adapter_id
       OR v_work.job_doc->>'intent_id' <> p_intent_id
       OR v_work.job_doc->>'approval_kind' <>
          'provenance_precheck_passed'
       OR NOT public.research_lab_source_add_provenance_leg1_authority_matches_v1(
           v_intent.submission_id,
           v_intent.adapter_id,
           v_intent.miner_hotkey,
           v_intent.provenance_receipt_hash,
           v_intent.provenance_artifact_hash
       ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance reward intent scope differs';
    END IF;

    SELECT reward_ref INTO v_existing_reward_ref
    FROM public.research_lab_source_add_reward_obligations
    WHERE adapter_id = v_intent.adapter_id
      AND leg = 1;
    IF FOUND THEN
        UPDATE public.research_lab_source_add_reward_slots
        SET slot_status = 'released',
            updated_at = NOW()
        WHERE intent_id = p_intent_id
          AND slot_status = 'reserved';
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'finalized',
            reward_ref = v_existing_reward_ref,
            updated_at = NOW()
        WHERE intent_id = p_intent_id;
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'completed',
            result_doc = pg_catalog.jsonb_build_object(
                'status', 'already_created',
                'reward_ref', v_existing_reward_ref
            ),
            completed_at = NOW(),
            lease_token = NULL,
            leased_by = '',
            lease_expires_at = NULL,
            updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'already_created',
            'reward_ref', v_existing_reward_ref
        );
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-leg1-day:' || v_day::TEXT,
            0
        )
    );
    UPDATE public.research_lab_source_add_reward_slots
    SET slot_status = 'expired',
        updated_at = NOW()
    WHERE slot_status = 'reserved'
      AND (lease_expires_at <= NOW() OR slot_day <> v_day);

    SELECT * INTO v_existing_slot
    FROM public.research_lab_source_add_reward_slots
    WHERE intent_id = p_intent_id
      AND slot_day = v_day
      AND slot_status = 'reserved'
      AND lease_expires_at > NOW()
    FOR UPDATE;
    IF FOUND THEN
        IF v_existing_slot.approval_kind <>
           'provenance_precheck_passed' THEN
            RAISE EXCEPTION
                'SOURCE_ADD provenance reward slot kind differs';
        END IF;
        UPDATE public.research_lab_source_add_reward_slots
        SET work_id = p_work_id,
            lease_token = v_token,
            lease_expires_at = NOW() +
                pg_catalog.make_interval(secs => p_slot_lease_seconds),
            updated_at = NOW()
        WHERE slot_id = v_existing_slot.slot_id;
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'leased',
            updated_at = NOW()
        WHERE intent_id = p_intent_id;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'reserved',
            'slot_day', v_day,
            'slot_number', v_existing_slot.slot_number,
            'slot_lease_token', v_token,
            'lease_expires_at', NOW() +
                pg_catalog.make_interval(secs => p_slot_lease_seconds)
        );
    END IF;

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
      AND candidate_intent.intent_status IN (
          'queued', 'leased', 'retry_wait'
      )
      AND candidate_intent.available_at <= NOW()
      AND NOT EXISTS (
          SELECT 1
          FROM public.research_lab_source_add_reward_obligations existing
          WHERE existing.adapter_id = candidate.adapter_id
            AND existing.leg = 1
      )
    ORDER BY
        candidate.priority ASC,
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

    SELECT COUNT(*) INTO v_created
    FROM public.research_lab_source_add_reward_events
    WHERE reason IN (
        'leg1_provenance_precheck_passed',
        'leg1_functional_probe_passed'
    )
      AND created_at >= (v_day::TIMESTAMP AT TIME ZONE 'UTC')
      AND created_at < ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC');
    SELECT COUNT(*) INTO v_reserved
    FROM public.research_lab_source_add_reward_slots
    WHERE slot_day = v_day
      AND slot_status = 'reserved'
      AND lease_expires_at > NOW();
    IF v_created + v_reserved >= 50 THEN
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'retry_wait',
            available_at = ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
            updated_at = NOW()
        WHERE intent_id = p_intent_id;
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'retry_wait',
            available_at = ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
            lease_token = NULL,
            leased_by = '',
            lease_expires_at = NULL,
            result_doc = pg_catalog.jsonb_build_object(
                'status', 'daily_cap_fifo'
            ),
            updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN pg_catalog.jsonb_build_object(
            'status', 'daily_cap_fifo',
            'available_at', ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC')
        );
    END IF;

    SELECT number INTO v_slot
    FROM pg_catalog.generate_series(1, 50) number
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_slots slot
        WHERE slot.slot_day = v_day
          AND slot.slot_number = number
          AND slot.slot_status IN ('reserved', 'finalized')
    )
    ORDER BY number
    LIMIT 1;
    IF v_slot IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD reward slot accounting differs';
    END IF;

    INSERT INTO public.research_lab_source_add_reward_slots (
        slot_day,
        slot_number,
        intent_id,
        work_id,
        slot_status,
        lease_token,
        lease_expires_at,
        approval_kind
    ) VALUES (
        v_day,
        v_slot,
        p_intent_id,
        p_work_id,
        'reserved',
        v_token,
        NOW() + pg_catalog.make_interval(secs => p_slot_lease_seconds),
        'provenance_precheck_passed'
    );
    UPDATE public.research_lab_source_add_reward_intents
    SET intent_status = 'leased',
        updated_at = NOW()
    WHERE intent_id = p_intent_id;
    RETURN pg_catalog.jsonb_build_object(
        'status', 'reserved',
        'slot_day', v_day,
        'slot_number', v_slot,
        'slot_lease_token', v_token,
        'lease_expires_at', NOW() +
            pg_catalog.make_interval(secs => p_slot_lease_seconds)
    );
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_reserve_leg1_slot_v4(
        TEXT, TEXT, UUID, INTEGER, INTEGER
    ) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_reserve_leg1_slot_v4(
        TEXT, TEXT, UUID, INTEGER, INTEGER
    ) TO service_role;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_finalize_leg1_v4(
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
    v_intent public.research_lab_source_add_reward_intents%ROWTYPE;
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_slot public.research_lab_source_add_reward_slots%ROWTYPE;
    v_authority RECORD;
    v_decision RECORD;
    v_expected_trigger JSONB;
    v_day DATE;
    v_created INTEGER;
    v_existing_reward_ref TEXT;
BEGIN
    -- p_daily_cap and p_submission_doc remain wire-compatible only. The
    -- database owns 50/day. Reward state is kept in the intent, slot,
    -- obligation, and reward event without overwriting the independent
    -- submission/provisioning lifecycle stage.
    IF pg_catalog.jsonb_typeof(p_reward) <> 'object'
       OR pg_catalog.jsonb_typeof(p_submission_doc) <> 'object'
       OR p_reward->>'reward_ref' !~ '^source_add_reward:[0-9a-f]{16}$'
       OR p_reward->>'reward_kind' <> 'source_acceptance'
       OR COALESCE((p_reward->>'alpha_percent')::NUMERIC, 0) <> 0.2
       OR COALESCE((p_reward->>'reward_epochs')::INTEGER, 0) <> 20
       OR COALESCE((p_reward->>'start_epoch')::INTEGER, -1) < 0
       OR p_reward->>'state' <> 'active'
       OR p_reward->>'decision_receipt_hash'
          !~ '^sha256:[0-9a-f]{64}$'
       OR p_reward->>'decision_artifact_hash'
          !~ '^sha256:[0-9a-f]{64}$' THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance Leg 1 reward input is invalid';
    END IF;

    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND
       OR v_work.work_status <> 'leased'
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
    IF v_intent.approval_kind <> 'provenance_precheck_passed'
       OR v_intent.intent_status <> 'leased'
       OR v_work.submission_id <> v_intent.submission_id
       OR v_work.adapter_id <> v_intent.adapter_id
       OR v_work.job_doc->>'intent_id' <> v_intent.intent_id
       OR v_work.job_doc->>'approval_kind' <>
          'provenance_precheck_passed' THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance reward intent scope differs';
    END IF;

    SELECT * INTO v_slot
    FROM public.research_lab_source_add_reward_slots
    WHERE intent_id = p_intent_id
      AND slot_status = 'reserved'
      AND lease_token = p_slot_lease_token
    FOR UPDATE;
    IF NOT FOUND
       OR v_slot.work_id <> p_work_id
       OR v_slot.approval_kind <> 'provenance_precheck_passed'
       OR v_slot.lease_expires_at <= NOW() THEN
        RETURN pg_catalog.jsonb_build_object('status', 'slot_lost');
    END IF;
    v_day := v_slot.slot_day;
    IF v_day <> (NOW() AT TIME ZONE 'UTC')::DATE THEN
        UPDATE public.research_lab_source_add_reward_slots
        SET slot_status = 'released',
            updated_at = NOW()
        WHERE slot_id = v_slot.slot_id;
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'retry_wait',
            available_at = NOW(),
            updated_at = NOW()
        WHERE intent_id = p_intent_id;
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'retry_wait',
            available_at = NOW(),
            lease_token = NULL,
            leased_by = '',
            lease_expires_at = NULL,
            updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN pg_catalog.jsonb_build_object('status', 'slot_day_rolled');
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-leg1-day:' || v_day::TEXT,
            0
        )
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
            'source-add-leg1-adapter:' || v_intent.adapter_id,
            0
        )
    );
    SELECT * INTO v_authority
    FROM public.research_lab_source_add_provenance_leg1_authority_v1
    WHERE submission_id = v_intent.submission_id;
    IF NOT FOUND
       OR v_authority.adapter_id <> v_intent.adapter_id
       OR v_authority.miner_hotkey <> v_intent.miner_hotkey
       OR v_authority.provenance_receipt_hash <>
          v_intent.provenance_receipt_hash
       OR v_authority.provenance_artifact_hash <>
          v_intent.provenance_artifact_hash THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance Leg 1 authority differs';
    END IF;
    v_expected_trigger := pg_catalog.jsonb_build_object(
        'provenance_precheck_passed', TRUE,
        'submission_id', v_intent.submission_id,
        'precheck_status', 'provenance_precheck_passed',
        'provenance_receipt_hash', v_intent.provenance_receipt_hash,
        'provenance_artifact_hash', v_intent.provenance_artifact_hash,
        'provenance_result_hash', v_intent.provenance_artifact_hash
    );
    IF p_reward->'trigger_evidence_doc' IS DISTINCT FROM
       v_expected_trigger THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance Leg 1 evidence differs';
    END IF;

    SELECT
        receipt.receipt_hash,
        receipt.output_root,
        receipt.receipt_doc,
        link.artifact_hash
    INTO v_decision
    FROM public.research_lab_attested_execution_receipts_v2 receipt
    JOIN public.research_lab_attested_business_artifact_links_v2 link
      ON link.receipt_hash = receipt.receipt_hash
    WHERE receipt.receipt_hash = p_reward->>'decision_receipt_hash'
      AND receipt.role = 'gateway_coordinator'
      AND receipt.purpose = 'research_lab.reward_decision.v2'
      AND receipt.receipt_status = 'succeeded'
      AND receipt.output_root = p_reward->>'decision_artifact_hash'
      AND link.artifact_kind = 'source_add_reward_decision'
      AND link.artifact_ref = p_reward->>'reward_ref'
      AND link.artifact_hash = p_reward->>'decision_artifact_hash';
    IF NOT FOUND
       OR v_decision.output_root <> v_decision.artifact_hash
       OR v_decision.receipt_doc->'parent_receipt_hashes'
          IS DISTINCT FROM pg_catalog.jsonb_build_array(
              v_intent.provenance_receipt_hash
          )
       OR (
           SELECT COUNT(*)
           FROM public.research_lab_attested_receipt_edges_v2 edge
           WHERE edge.child_receipt_hash = v_decision.receipt_hash
       ) <> 1
       OR NOT EXISTS (
           SELECT 1
           FROM public.research_lab_attested_receipt_edges_v2 edge
           WHERE edge.child_receipt_hash = v_decision.receipt_hash
             AND edge.parent_receipt_hash =
                 v_intent.provenance_receipt_hash
       ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance Leg 1 decision ancestry differs';
    END IF;

    SELECT reward_ref INTO v_existing_reward_ref
    FROM public.research_lab_source_add_reward_obligations
    WHERE adapter_id = v_intent.adapter_id
      AND leg = 1;
    IF NOT FOUND THEN
        SELECT COUNT(*) INTO v_created
        FROM public.research_lab_source_add_reward_events
        WHERE reason IN (
            'leg1_provenance_precheck_passed',
            'leg1_functional_probe_passed'
        )
          AND created_at >= (v_day::TIMESTAMP AT TIME ZONE 'UTC')
          AND created_at < ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC');
        IF v_created >= 50 THEN
            UPDATE public.research_lab_source_add_reward_slots
            SET slot_status = 'released',
                updated_at = NOW()
            WHERE slot_id = v_slot.slot_id;
            UPDATE public.research_lab_source_add_reward_intents
            SET intent_status = 'retry_wait',
                available_at =
                    ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
                updated_at = NOW()
            WHERE intent_id = p_intent_id;
            UPDATE public.research_lab_source_add_work_items
            SET work_status = 'retry_wait',
                available_at =
                    ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
                lease_token = NULL,
                leased_by = '',
                lease_expires_at = NULL,
                updated_at = NOW()
            WHERE work_id = p_work_id;
            RETURN pg_catalog.jsonb_build_object(
                'status', 'daily_cap_fifo'
            );
        END IF;

        INSERT INTO public.research_lab_source_add_reward_obligations (
            reward_ref,
            adapter_id,
            catalog_id,
            miner_hotkey,
            leg,
            reward_kind,
            alpha_percent,
            reward_epochs,
            start_epoch,
            trigger_evidence_doc,
            public_label
        ) VALUES (
            p_reward->>'reward_ref',
            v_intent.adapter_id,
            NULL,
            v_intent.miner_hotkey,
            1,
            'source_acceptance',
            0.2,
            20,
            (p_reward->>'start_epoch')::INTEGER,
            v_expected_trigger,
            'Source acceptance reward'
        );
        INSERT INTO public.research_lab_source_add_reward_events (
            reward_ref,
            seq,
            reward_status,
            reason
        ) VALUES (
            p_reward->>'reward_ref',
            0,
            'active',
            'leg1_provenance_precheck_passed'
        );
        v_existing_reward_ref := p_reward->>'reward_ref';
    END IF;

    UPDATE public.research_lab_source_add_reward_slots
    SET slot_status = 'finalized',
        reward_ref = v_existing_reward_ref,
        updated_at = NOW()
    WHERE slot_id = v_slot.slot_id;
    UPDATE public.research_lab_source_add_reward_intents
    SET intent_status = 'finalized',
        reward_ref = v_existing_reward_ref,
        updated_at = NOW()
    WHERE intent_id = p_intent_id;
    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'completed',
        result_doc = pg_catalog.jsonb_build_object(
            'status', 'created',
            'reward_ref', v_existing_reward_ref
        ),
        completed_at = NOW(),
        lease_token = NULL,
        leased_by = '',
        lease_expires_at = NULL,
        updated_at = NOW()
    WHERE work_id = p_work_id;

    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_submission_current current
        WHERE current.submission_id = v_intent.submission_id
          AND current.adapter_id = v_intent.adapter_id
          AND current.miner_hotkey = v_intent.miner_hotkey
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance submission disappeared';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
        'status', 'created',
        'reward_ref', v_existing_reward_ref
    );
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_leg1_v4(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_leg1_v4(
    TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB
) TO service_role;

-- Backfill every historical row which has the exact receipt and business-link
-- authority. Rows from older releases without that durable authority remain
-- excluded rather than reconstructing or fabricating an attestation.
DO $$
DECLARE
    v_result JSONB;
    v_authority_count INTEGER;
BEGIN
    v_result :=
        public.research_lab_source_add_reconcile_provenance_leg1_v1();
    SELECT COUNT(*)::INTEGER INTO v_authority_count
    FROM public.research_lab_source_add_provenance_leg1_authority_v1 authority
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_reward_obligations paid
        WHERE paid.adapter_id = authority.adapter_id
          AND paid.leg = 1
    );
    IF COALESCE((v_result->>'eligible_count')::INTEGER, -1) <>
       v_authority_count THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance Leg 1 backfill count differs';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_provenance_leg1_authority_v1
            authority
        WHERE NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_reward_obligations paid
            WHERE paid.adapter_id = authority.adapter_id
              AND paid.leg = 1
        )
          AND NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_reward_intents intent
            WHERE intent.submission_id = authority.submission_id
              AND intent.adapter_id = authority.adapter_id
              AND intent.miner_hotkey = authority.miner_hotkey
              AND intent.leg = 1
              AND (
                  intent.approval_kind = 'post_accept_functional_probe'
                  OR (
                      intent.approval_kind =
                          'provenance_precheck_passed'
                      AND intent.provenance_receipt_hash =
                          authority.provenance_receipt_hash
                      AND intent.provenance_artifact_hash =
                          authority.provenance_artifact_hash
                      AND (
                          EXISTS (
                              SELECT 1
                              FROM public.research_lab_source_add_work_items work
                              WHERE work.submission_id = authority.submission_id
                                AND work.adapter_id = authority.adapter_id
                                AND work.work_kind = 'leg1_reward'
                                AND work.job_doc->>'intent_id' = intent.intent_id
                          )
                          OR EXISTS (
                              SELECT 1
                              FROM public.research_lab_source_add_reward_obligations
                                  reward
                              WHERE reward.adapter_id = authority.adapter_id
                                AND reward.leg = 1
                          )
                      )
                  )
              )
        )
    ) THEN
        RAISE EXCEPTION
            'SOURCE_ADD provenance Leg 1 backfill is incomplete';
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION
    public.research_lab_source_add_post_accept_leg1_contract_v3()
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_service_role_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_roles
        WHERE rolname = 'service_role'
    ) INTO v_service_role_exists;
    RETURN pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.source_add_post_accept_leg1_contract.v3',
        'daily_cap', 50,
        'leg1_alpha_percent', 0.2,
        'leg1_reward_epochs', 20,
        'approval_boundary', 'provenance_precheck_passed',
        'backfill_policy', 'all_exact_attested_provenance',
        'public_trigger_fields', pg_catalog.jsonb_build_array(
            'precheck_status',
            'provenance_artifact_hash',
            'provenance_precheck_passed',
            'provenance_receipt_hash',
            'provenance_result_hash',
            'submission_id'
        ),
        'authority_view',
            'research_lab_source_add_provenance_leg1_authority_v1',
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
                                    'return_type',
                                        proc.prorettype::REGTYPE::TEXT
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
                        'configure_probe_v3',
                        'public.research_lab_source_add_configure_probe_v3(text,text,jsonb,jsonb,text,text,text)'
                    ),
                    (
                        'contract_v2',
                        'public.research_lab_source_add_post_accept_leg1_contract_v2()'
                    ),
                    (
                        'contract_v3',
                        'public.research_lab_source_add_post_accept_leg1_contract_v3()'
                    ),
                    (
                        'enqueue_leg1_after_provenance_v1',
                        'public.research_lab_source_add_enqueue_leg1_after_provenance_v1(text,text,text,text,text)'
                    ),
                    (
                        'enqueue_provenance_trigger_v1',
                        'public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1()'
                    ),
                    (
                        'enqueue_provision_smoke_v2',
                        'public.research_lab_source_add_enqueue_provision_smoke_v2(text,text,text,text,jsonb,jsonb)'
                    ),
                    (
                        'finalize_leg1_v4',
                        'public.research_lab_source_add_finalize_leg1_v4(text,text,uuid,uuid,integer,jsonb,jsonb)'
                    ),
                    (
                        'finalize_provision_smoke_v3',
                        'public.research_lab_source_add_finalize_provision_smoke_v3(text,uuid,text,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'finalize_provision_v3',
                        'public.research_lab_source_add_finalize_provision_v3(text,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'provenance_authority_matches_v1',
                        'public.research_lab_source_add_provenance_leg1_authority_matches_v1(text,text,text,text,text)'
                    ),
                    (
                        'reject_current_builtin_v3',
                        'public.research_lab_source_add_reject_current_builtin_v3(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)'
                    ),
                    (
                        'reconcile_provenance_leg1_v1',
                        'public.research_lab_source_add_reconcile_provenance_leg1_v1()'
                    ),
                    (
                        'reserve_leg1_slot_v4',
                        'public.research_lab_source_add_reserve_leg1_slot_v4(text,text,uuid,integer,integer)'
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
                        'trigger_eligible_v3',
                        'public.enforce_research_lab_source_add_eligible_v3()'
                    ),
                    (
                        'trigger_leg1_initial_event_v2',
                        'public.enforce_research_lab_source_add_leg1_initial_event_v2()'
                    ),
                    (
                        'trigger_leg1_initial_event_v3',
                        'public.enforce_research_lab_source_add_leg1_initial_event_v3()'
                    ),
                    (
                        'trigger_leg1_obligation_v2',
                        'public.enforce_research_lab_source_add_leg1_obligation_v2()'
                    ),
                    (
                        'trigger_leg1_obligation_v3',
                        'public.enforce_research_lab_source_add_leg1_obligation_v3()'
                    ),
                    (
                        'trigger_leg1_slot_v2',
                        'public.enforce_research_lab_source_add_leg1_slot_v2()'
                    ),
                    (
                        'trigger_leg1_slot_v3',
                        'public.enforce_research_lab_source_add_leg1_slot_v3()'
                    ),
                    (
                        'trigger_leg1_work_v2',
                        'public.enforce_research_lab_source_add_leg1_work_v2()'
                    ),
                    (
                        'trigger_leg1_work_v3',
                        'public.enforce_research_lab_source_add_leg1_work_v3()'
                    )
            ) AS authority(name, signature)
            LEFT JOIN pg_catalog.pg_proc proc
              ON proc.oid = pg_catalog.to_regprocedure(authority.signature)
            LEFT JOIN pg_catalog.pg_language language
              ON language.oid = proc.prolang
        ),
        'trigger_authority_sha256', (
            SELECT 'sha256:' || pg_catalog.encode(
                extensions.digest(
                    pg_catalog.convert_to(
                        COALESCE(
                            pg_catalog.jsonb_object_agg(
                                expected.trigger_name,
                                pg_catalog.jsonb_build_object(
                                    'definition', pg_catalog.pg_get_triggerdef(
                                        trigger_row.oid,
                                        TRUE
                                    ),
                                    'enabled', trigger_row.tgenabled
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
                    ('trg_source_add_acceptance_v2'),
                    ('trg_source_add_eligible_v2'),
                    ('trg_source_add_eligible_v3'),
                    ('trg_source_add_enqueue_provenance_leg1_v1'),
                    ('trg_source_add_leg1_initial_event_v2'),
                    ('trg_source_add_leg1_initial_event_v3'),
                    ('trg_source_add_leg1_obligation_v2'),
                    ('trg_source_add_leg1_obligation_v3'),
                    ('trg_source_add_leg1_slot_v2'),
                    ('trg_source_add_leg1_slot_v3'),
                    ('trg_source_add_leg1_work_v2'),
                    ('trg_source_add_leg1_work_v3')
            ) AS expected(trigger_name)
            LEFT JOIN pg_catalog.pg_trigger trigger_row
              ON trigger_row.tgname = expected.trigger_name
             AND NOT trigger_row.tgisinternal
        ),
        'view_authority_sha256', 'sha256:' || pg_catalog.encode(
            extensions.digest(
                pg_catalog.convert_to(
                    COALESCE(
                        pg_catalog.pg_get_viewdef(
                            pg_catalog.to_regclass(
                                'public.research_lab_source_add_provenance_leg1_authority_v1'
                            ),
                            TRUE
                        ),
                        ''
                    ),
                    'UTF8'
                ),
                'sha256'
            ),
            'hex'
        ),
        'functions', pg_catalog.jsonb_build_object(
            'configure_probe_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_configure_probe_v3(text,text,jsonb,jsonb,text,text,text)'
            ) IS NOT NULL,
            'enqueue_leg1_after_provenance_v1',
                pg_catalog.to_regprocedure(
                    'public.research_lab_source_add_enqueue_leg1_after_provenance_v1(text,text,text,text,text)'
                ) IS NOT NULL,
            'enqueue_provision_smoke_v2', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_enqueue_provision_smoke_v2(text,text,text,text,jsonb,jsonb)'
            ) IS NOT NULL,
            'finalize_leg1_v4', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_leg1_v4(text,text,uuid,uuid,integer,jsonb,jsonb)'
            ) IS NOT NULL,
            'finalize_provision_smoke_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_provision_smoke_v3(text,uuid,text,jsonb,jsonb,jsonb)'
            ) IS NOT NULL,
            'finalize_provision_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_finalize_provision_v3(text,jsonb,jsonb,jsonb)'
            ) IS NOT NULL,
            'reject_current_builtin_v3', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_reject_current_builtin_v3(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)'
            ) IS NOT NULL,
            'reconcile_provenance_leg1_v1',
                pg_catalog.to_regprocedure(
                    'public.research_lab_source_add_reconcile_provenance_leg1_v1()'
                ) IS NOT NULL,
            'reserve_leg1_slot_v4', pg_catalog.to_regprocedure(
                'public.research_lab_source_add_reserve_leg1_slot_v4(text,text,uuid,integer,integer)'
            ) IS NOT NULL
        ),
        'triggers', pg_catalog.jsonb_build_object(
            'automatic_enqueue', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname =
                      'trg_source_add_enqueue_provenance_leg1_v1'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 5
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'eligible_v2', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname = 'trg_source_add_eligible_v2'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_eligible_v2()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'eligible_v3', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname = 'trg_source_add_eligible_v3'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_eligible_v3()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_initial_event_v3', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname =
                      'trg_source_add_leg1_initial_event_v3'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_initial_event_v3()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_obligation_v3', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname =
                      'trg_source_add_leg1_obligation_v3'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_obligation_v3()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_slot_v3', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname = 'trg_source_add_leg1_slot_v3'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 23
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_slot_v3()'
                  )
                  AND NOT trigger_row.tgisinternal
            ),
            'leg1_work_v3', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_trigger trigger_row
                WHERE trigger_row.tgname = 'trg_source_add_leg1_work_v3'
                  AND trigger_row.tgenabled = 'O'
                  AND trigger_row.tgtype = 7
                  AND trigger_row.tgfoid = pg_catalog.to_regprocedure(
                      'public.enforce_research_lab_source_add_leg1_work_v3()'
                  )
                  AND NOT trigger_row.tgisinternal
            )
        ),
        'columns', pg_catalog.jsonb_build_object(
            'intent_approval_kind', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_attribute attribute
                WHERE attribute.attrelid = pg_catalog.to_regclass(
                    'public.research_lab_source_add_reward_intents'
                )
                  AND attribute.attname = 'approval_kind'
                  AND attribute.attnotnull
                  AND NOT attribute.attisdropped
            ),
            'intent_provenance_artifact_hash', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_attribute attribute
                WHERE attribute.attrelid = pg_catalog.to_regclass(
                    'public.research_lab_source_add_reward_intents'
                )
                  AND attribute.attname = 'provenance_artifact_hash'
                  AND attribute.attnotnull
                  AND NOT attribute.attisdropped
            ),
            'intent_provenance_receipt_hash', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_attribute attribute
                WHERE attribute.attrelid = pg_catalog.to_regclass(
                    'public.research_lab_source_add_reward_intents'
                )
                  AND attribute.attname = 'provenance_receipt_hash'
                  AND attribute.attnotnull
                  AND NOT attribute.attisdropped
            ),
            'slot_approval_kind', EXISTS (
                SELECT 1
                FROM pg_catalog.pg_attribute attribute
                WHERE attribute.attrelid = pg_catalog.to_regclass(
                    'public.research_lab_source_add_reward_slots'
                )
                  AND attribute.attname = 'approval_kind'
                  AND attribute.attnotnull
                  AND NOT attribute.attisdropped
            )
        ),
        'permissions', pg_catalog.jsonb_build_object(
            'service_role_exists', v_service_role_exists,
            'candidate_callable', v_service_role_exists
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_enqueue_leg1_after_provenance_v1(text,text,text,text,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reconcile_provenance_leg1_v1()',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_configure_probe_v3(text,text,jsonb,jsonb,text,text,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_enqueue_provision_smoke_v2(text,text,text,text,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_v3(text,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_smoke_v3(text,uuid,text,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reject_current_builtin_v3(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reserve_leg1_slot_v4(text,text,uuid,integer,integer)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_leg1_v4(text,text,uuid,uuid,integer,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_table_privilege(
                    'service_role',
                    'public.research_lab_source_add_provenance_leg1_authority_v1',
                    'SELECT'
                ),
            'internal_not_callable', v_service_role_exists
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_provenance_leg1_authority_matches_v1(text,text,text,text,text)',
                    'EXECUTE'
                )
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_enqueue_provenance_leg1_trigger_v1()',
                    'EXECUTE'
                )
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.enforce_research_lab_source_add_eligible_v3()',
                    'EXECUTE'
                )
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.enforce_research_lab_source_add_leg1_work_v3()',
                    'EXECUTE'
                )
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.enforce_research_lab_source_add_leg1_slot_v3()',
                    'EXECUTE'
                )
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.enforce_research_lab_source_add_leg1_obligation_v3()',
                    'EXECUTE'
                )
                AND NOT pg_catalog.has_function_privilege(
                    'service_role',
                    'public.enforce_research_lab_source_add_leg1_initial_event_v3()',
                    'EXECUTE'
                ),
            'rollback_v2_callable', v_service_role_exists
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_post_accept_leg1_contract_v2()',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_configure_probe_v2(text,text,jsonb,jsonb,text,text,text)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_v2(text,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_provision_smoke_v2(text,uuid,text,jsonb,jsonb,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reject_current_builtin_v2(text,uuid,text,jsonb,text,jsonb,jsonb,jsonb,jsonb)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_reserve_leg1_slot_v3(text,text,uuid,integer,integer)',
                    'EXECUTE'
                )
                AND pg_catalog.has_function_privilege(
                    'service_role',
                    'public.research_lab_source_add_finalize_leg1_v3(text,text,uuid,uuid,integer,jsonb,jsonb)',
                    'EXECUTE'
                )
        )
    );
END;
$$;

REVOKE ALL ON FUNCTION
    public.research_lab_source_add_post_accept_leg1_contract_v3()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_source_add_post_accept_leg1_contract_v3()
    TO service_role;

COMMENT ON TABLE public.research_lab_source_add_reward_obligations IS
    'Append-only SOURCE_ADD reward legs: each exact attested credible provenance pass may create one 0.2% Leg 1 obligation; catalog qualification and enabled implementation riders are separate later decisions.';

NOTIFY pgrst, 'reload schema';

COMMIT;
