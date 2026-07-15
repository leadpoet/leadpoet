-- SOURCE_ADD measured functional workflow, atomic admission, and FIFO Leg 1.
--
-- Additive only. Existing SOURCE_ADD rewards/catalog/provisioning history is
-- preserved. Operational queue/lease tables are mutable by service-role RPCs;
-- business transitions, probe attempts, and identity history remain append-only.

BEGIN;

SET LOCAL lock_timeout = '5s';

ALTER TABLE public.research_lab_source_add_submissions
    ADD COLUMN IF NOT EXISTS source_identity_version TEXT NOT NULL DEFAULT 'v1';

ALTER TABLE public.research_lab_source_add_submissions
    DROP CONSTRAINT IF EXISTS research_lab_source_add_submissions_identity_version_check;
ALTER TABLE public.research_lab_source_add_submissions
    ADD CONSTRAINT research_lab_source_add_submissions_identity_version_check
    CHECK (source_identity_version IN ('v1', 'v2')) NOT VALID;
ALTER TABLE public.research_lab_source_add_submissions
    VALIDATE CONSTRAINT research_lab_source_add_submissions_identity_version_check;

ALTER TABLE public.research_lab_source_add_submissions
    DROP CONSTRAINT IF EXISTS research_lab_source_add_submissions_stage_check;
ALTER TABLE public.research_lab_source_add_submissions
    ADD CONSTRAINT research_lab_source_add_submissions_stage_check
    CHECK (stage IN (
        'submitted',
        'manifest_validated',
        'provenance_queued',
        'provenance_precheck_passed',
        'needs_manual_review',
        'rejected_precheck',
        'functional_probe_queued',
        'awaiting_operator_credential',
        'functional_probe_retryable',
        'functional_probe_passed',
        'functional_probe_failed',
        'leg1_queued',
        'leg1_created',
        'static_scan_passed',
        'llm_review_passed',
        'trial_completed',
        'accepted',
        'rejected'
    )) NOT VALID;
ALTER TABLE public.research_lab_source_add_submissions
    VALIDATE CONSTRAINT research_lab_source_add_submissions_stage_check;

CREATE OR REPLACE VIEW public.research_lab_source_add_submission_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (submission_id)
    submission_row_id,
    submission_id,
    schema_version,
    adapter_id,
    miner_hotkey,
    stage,
    seq,
    measured_trial_yield,
    submission_doc,
    created_at,
    precheck_status,
    precheck_doc,
    source_identity_hash,
    source_identity_version
FROM public.research_lab_source_add_submissions
ORDER BY submission_id, seq DESC, created_at DESC;

REVOKE ALL ON TABLE public.research_lab_source_add_submission_current FROM PUBLIC, anon, authenticated;
GRANT SELECT ON TABLE public.research_lab_source_add_submission_current TO service_role;

CREATE OR REPLACE FUNCTION public.prevent_research_lab_source_add_history_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$;

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_identity_events (
    identity_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_version TEXT NOT NULL CHECK (identity_version IN ('v1', 'v2')),
    source_identity_hash TEXT NOT NULL CHECK (source_identity_hash ~ '^sha256:[0-9a-f]{64}$'),
    submission_id TEXT NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    reservation_status TEXT NOT NULL CHECK (reservation_status IN ('reserved', 'released')),
    seq INTEGER NOT NULL CHECK (seq >= 0),
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (identity_version, source_identity_hash, seq)
);

CREATE INDEX IF NOT EXISTS idx_source_add_identity_submission
    ON public.research_lab_source_add_identity_events (submission_id, created_at DESC);

CREATE OR REPLACE VIEW public.research_lab_source_add_identity_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (identity_version, source_identity_hash)
    identity_event_id,
    identity_version,
    source_identity_hash,
    submission_id,
    adapter_id,
    miner_hotkey,
    reservation_status,
    seq,
    reason,
    created_at
FROM public.research_lab_source_add_identity_events
ORDER BY identity_version, source_identity_hash, seq DESC, created_at DESC;

INSERT INTO public.research_lab_source_add_identity_events (
    identity_version,
    source_identity_hash,
    submission_id,
    adapter_id,
    miner_hotkey,
    reservation_status,
    seq,
    reason
)
SELECT DISTINCT ON (s.source_identity_hash)
    'v1',
    s.source_identity_hash,
    s.submission_id,
    s.adapter_id,
    s.miner_hotkey,
    'reserved',
    0,
    'migration_96_existing_nonterminal'
FROM public.research_lab_source_add_submission_current s
WHERE s.source_identity_hash ~ '^sha256:[0-9a-f]{64}$'
  AND s.stage NOT IN ('rejected', 'rejected_precheck', 'functional_probe_failed')
ORDER BY s.source_identity_hash, s.created_at ASC
ON CONFLICT (identity_version, source_identity_hash, seq) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_probe_config_events (
    config_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_ref TEXT NOT NULL CHECK (config_ref ~ '^source_add_probe_config:[0-9a-f]{16}$'),
    submission_id TEXT NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id TEXT NOT NULL,
    config_status TEXT NOT NULL CHECK (config_status IN ('active', 'disabled')),
    seq INTEGER NOT NULL CHECK (seq >= 0),
    probe_doc JSONB NOT NULL CHECK (
        jsonb_typeof(probe_doc) = 'object'
        AND probe_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
    ),
    credential_envelope JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(credential_envelope) = 'object'
        AND credential_envelope::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:)'
    ),
    actor_ref TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (submission_id, seq),
    UNIQUE (config_ref)
);

CREATE OR REPLACE VIEW public.research_lab_source_add_probe_config_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (submission_id)
    config_event_id,
    config_ref,
    submission_id,
    adapter_id,
    config_status,
    seq,
    probe_doc,
    credential_envelope,
    actor_ref,
    created_at
FROM public.research_lab_source_add_probe_config_events
ORDER BY submission_id, seq DESC, created_at DESC;

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_functional_probe_attempts (
    attempt_ref TEXT PRIMARY KEY CHECK (attempt_ref ~ '^source_add_probe_attempt:[0-9a-f]{16}$'),
    submission_id TEXT NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id TEXT NOT NULL,
    work_id TEXT NOT NULL CHECK (work_id ~ '^source_add_work:[0-9a-f]{16}$'),
    attempt_number INTEGER NOT NULL CHECK (attempt_number BETWEEN 1 AND 20),
    evaluation_mode TEXT NOT NULL DEFAULT 'functional_probe' CHECK (
        evaluation_mode IN ('functional_probe', 'provisioning_smoke')
    ),
    config_ref TEXT NOT NULL CHECK (config_ref ~ '^source_add_probe_config:[0-9a-f]{16}$'),
    result_status TEXT NOT NULL CHECK (result_status IN (
        'passed', 'retryable', 'awaiting_operator', 'manual_review', 'failed'
    )),
    route_hash TEXT NOT NULL DEFAULT '' CHECK (route_hash = '' OR route_hash ~ '^sha256:[0-9a-f]{64}$'),
    response_hash TEXT NOT NULL DEFAULT '' CHECK (response_hash = '' OR response_hash ~ '^sha256:[0-9a-f]{64}$'),
    status_class TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT '',
    byte_count INTEGER NOT NULL DEFAULT 0 CHECK (byte_count BETWEEN 0 AND 1048576),
    duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms BETWEEN 0 AND 180000),
    retry_after_seconds INTEGER NOT NULL DEFAULT 0 CHECK (retry_after_seconds BETWEEN 0 AND 21600),
    reason_codes JSONB NOT NULL DEFAULT '[]'::JSONB CHECK (jsonb_typeof(reason_codes) = 'array'),
    receipt_hash TEXT NOT NULL DEFAULT '' CHECK (receipt_hash = '' OR receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    business_artifact_hash TEXT NOT NULL DEFAULT '' CHECK (
        business_artifact_hash = '' OR business_artifact_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    result_doc JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(result_doc) = 'object'
        AND result_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|request_body|response_body|provider_output|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (work_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_source_add_functional_current
    ON public.research_lab_source_add_functional_probe_attempts (submission_id, created_at DESC);

CREATE OR REPLACE VIEW public.research_lab_source_add_functional_probe_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (submission_id)
    *
FROM public.research_lab_source_add_functional_probe_attempts
WHERE evaluation_mode = 'functional_probe'
ORDER BY submission_id, created_at DESC, attempt_number DESC, attempt_ref DESC;

CREATE OR REPLACE VIEW public.research_lab_source_add_provisioning_smoke_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (submission_id)
    *
FROM public.research_lab_source_add_functional_probe_attempts
WHERE evaluation_mode = 'provisioning_smoke'
ORDER BY submission_id, created_at DESC, attempt_number DESC, attempt_ref DESC;

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_work_items (
    work_id TEXT PRIMARY KEY CHECK (work_id ~ '^source_add_work:[0-9a-f]{16}$'),
    submission_id TEXT NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id TEXT NOT NULL,
    work_kind TEXT NOT NULL CHECK (work_kind IN (
        'provenance', 'functional_probe', 'leg1_reward', 'provisioning_smoke'
    )),
    work_status TEXT NOT NULL CHECK (work_status IN (
        'queued', 'leased', 'retry_wait', 'completed', 'dead_letter', 'cancelled'
    )),
    priority INTEGER NOT NULL DEFAULT 100 CHECK (priority BETWEEN 0 AND 1000),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count BETWEEN 0 AND 20),
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_token UUID,
    leased_by TEXT NOT NULL DEFAULT '',
    lease_expires_at TIMESTAMPTZ,
    job_doc JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(job_doc) = 'object'
        AND job_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
    ),
    result_doc JSONB NOT NULL DEFAULT '{}'::JSONB CHECK (
        jsonb_typeof(result_doc) = 'object'
        AND result_doc::TEXT !~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|request_body|response_body|provider_output|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE (submission_id, work_kind, work_id)
);

CREATE INDEX IF NOT EXISTS idx_source_add_work_claim
    ON public.research_lab_source_add_work_items (work_status, available_at, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_source_add_work_submission
    ON public.research_lab_source_add_work_items (submission_id, created_at DESC);

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_reward_intents (
    intent_id TEXT PRIMARY KEY CHECK (intent_id ~ '^source_add_reward_intent:[0-9a-f]{16}$'),
    submission_id TEXT NOT NULL CHECK (submission_id ~ '^source_add_submission:[0-9a-f]{16}$'),
    adapter_id TEXT NOT NULL,
    miner_hotkey TEXT NOT NULL,
    leg INTEGER NOT NULL DEFAULT 1 CHECK (leg = 1),
    intent_status TEXT NOT NULL CHECK (intent_status IN (
        'queued', 'leased', 'retry_wait', 'finalized', 'cancelled'
    )),
    functional_receipt_hash TEXT NOT NULL CHECK (functional_receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
    business_artifact_hash TEXT NOT NULL CHECK (business_artifact_hash ~ '^sha256:[0-9a-f]{64}$'),
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reward_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (adapter_id, leg)
);

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_reward_slots (
    slot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slot_day DATE NOT NULL,
    slot_number INTEGER NOT NULL CHECK (slot_number BETWEEN 1 AND 100),
    intent_id TEXT NOT NULL REFERENCES public.research_lab_source_add_reward_intents(intent_id) ON DELETE RESTRICT,
    work_id TEXT NOT NULL REFERENCES public.research_lab_source_add_work_items(work_id) ON DELETE RESTRICT,
    slot_status TEXT NOT NULL CHECK (slot_status IN ('reserved', 'finalized', 'expired', 'released')),
    lease_token UUID NOT NULL,
    lease_expires_at TIMESTAMPTZ NOT NULL,
    reward_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Expired/released rows remain as audit history and must not prevent a later
-- intent from reusing the UTC-day slot. Only live/finalized claims reserve it.
ALTER TABLE public.research_lab_source_add_reward_slots
    DROP CONSTRAINT IF EXISTS research_lab_source_add_reward_slots_slot_day_slot_number_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_source_add_reward_slots_live_day_slot
    ON public.research_lab_source_add_reward_slots (slot_day, slot_number)
    WHERE slot_status IN ('reserved', 'finalized');
CREATE UNIQUE INDEX IF NOT EXISTS idx_source_add_reward_slots_live_intent
    ON public.research_lab_source_add_reward_slots (intent_id)
    WHERE slot_status IN ('reserved', 'finalized');

CREATE TABLE IF NOT EXISTS public.research_lab_source_add_control (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    paused BOOLEAN NOT NULL DEFAULT TRUE,
    reason TEXT NOT NULL DEFAULT 'migration_96_disabled_by_default',
    actor_ref TEXT NOT NULL DEFAULT 'operator:migration',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO public.research_lab_source_add_control (singleton)
VALUES (TRUE)
ON CONFLICT (singleton) DO NOTHING;

DROP TRIGGER IF EXISTS trg_source_add_identity_no_mutation
    ON public.research_lab_source_add_identity_events;
CREATE TRIGGER trg_source_add_identity_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_add_identity_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_add_history_mutation();

DROP TRIGGER IF EXISTS trg_source_add_probe_config_no_mutation
    ON public.research_lab_source_add_probe_config_events;
CREATE TRIGGER trg_source_add_probe_config_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_add_probe_config_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_add_history_mutation();

DROP TRIGGER IF EXISTS trg_source_add_probe_attempt_no_mutation
    ON public.research_lab_source_add_functional_probe_attempts;
CREATE TRIGGER trg_source_add_probe_attempt_no_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_source_add_functional_probe_attempts
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_source_add_history_mutation();

DROP FUNCTION IF EXISTS public.research_lab_source_add_admit(
    JSONB, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER
);

CREATE OR REPLACE FUNCTION public.research_lab_source_add_admit(
    p_record_doc JSONB,
    p_identity_hash TEXT,
    p_documentation_identity_hash TEXT,
    p_legacy_identity_hash TEXT,
    p_work_id TEXT,
    p_max_open INTEGER,
    p_max_day INTEGER,
    p_max_30d INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_submission_id TEXT := p_record_doc->>'submission_id';
    v_adapter_id TEXT := p_record_doc->>'adapter_id';
    v_miner_hotkey TEXT := p_record_doc->>'miner_hotkey';
    v_existing RECORD;
    v_open INTEGER;
    v_day INTEGER;
    v_30d INTEGER;
    v_hash TEXT;
    v_identity RECORD;
    v_seq INTEGER;
    v_doc JSONB;
    v_existing_terminal BOOLEAN := FALSE;
    v_start_seq INTEGER := 0;
BEGIN
    IF v_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR v_adapter_id = '' OR v_miner_hotkey = ''
       OR p_identity_hash !~ '^sha256:[0-9a-f]{64}$'
       OR (COALESCE(p_documentation_identity_hash, '') <> ''
           AND p_documentation_identity_hash !~ '^sha256:[0-9a-f]{64}$')
       OR (COALESCE(p_legacy_identity_hash, '') <> ''
           AND p_legacy_identity_hash !~ '^sha256:[0-9a-f]{64}$')
       OR p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_max_open < 1 OR p_max_day < 1 OR p_max_30d < 1
       OR jsonb_typeof(p_record_doc) <> 'object'
       OR COALESCE(p_record_doc->'credential_envelope', '{}'::JSONB) <> '{}'::JSONB
       OR COALESCE(p_record_doc->'manifest'->>'credential_policy', '') <> 'no_credentials'
       OR COALESCE(p_record_doc->'manifest'->>'credential_ref', '') <> ''
       OR p_record_doc::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])' THEN
        RAISE EXCEPTION 'SOURCE_ADD admission input is invalid';
    END IF;

    SELECT * INTO v_existing
    FROM public.research_lab_source_add_submission_current
    WHERE submission_id = v_submission_id;
    IF FOUND THEN
        IF v_existing.stage NOT IN (
            'rejected', 'rejected_precheck', 'functional_probe_failed'
        ) THEN
            RETURN jsonb_build_object('status', 'duplicate');
        END IF;
        IF v_existing.adapter_id <> v_adapter_id
           OR v_existing.miner_hotkey <> v_miner_hotkey THEN
            RAISE EXCEPTION 'SOURCE_ADD terminal resubmission ownership differs';
        END IF;
        v_existing_terminal := TRUE;
        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_start_seq
        FROM public.research_lab_source_add_submissions
        WHERE submission_id = v_submission_id;
    END IF;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_source_add_work_items
        WHERE work_id = p_work_id
    ) THEN
        RETURN jsonb_build_object('status', 'duplicate');
    END IF;

    FOR v_hash IN
        SELECT DISTINCT item FROM unnest(ARRAY[
            p_identity_hash,
            NULLIF(NULLIF(p_documentation_identity_hash, ''), p_identity_hash),
            NULLIF(NULLIF(p_legacy_identity_hash, ''), p_identity_hash)
        ]) item
        WHERE item IS NOT NULL ORDER BY item
    LOOP
        PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('source-add-identity:' || v_hash, 0));
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_identity_current
        WHERE reservation_status = 'reserved'
          AND source_identity_hash IN (
              p_identity_hash,
              NULLIF(p_documentation_identity_hash, ''),
              NULLIF(p_legacy_identity_hash, '')
          )
    ) OR EXISTS (
        SELECT 1
        FROM public.research_lab_source_catalog
        WHERE source_identity_hash IN (
            p_identity_hash,
            NULLIF(p_documentation_identity_hash, ''),
            NULLIF(p_legacy_identity_hash, '')
        )
    ) THEN
        RETURN jsonb_build_object('status', 'duplicate');
    END IF;

    -- Identity locks protect duplicate privacy; this independent hotkey lock
    -- makes all three admission limits exact under concurrent unique sources.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-hotkey:' || v_miner_hotkey, 0)
    );

    SELECT COUNT(*) INTO v_open
    FROM public.research_lab_source_add_submission_current
    WHERE miner_hotkey = v_miner_hotkey
      AND stage IN (
        'submitted', 'manifest_validated', 'provenance_queued',
        'provenance_precheck_passed', 'needs_manual_review',
        'functional_probe_queued', 'awaiting_operator_credential',
        'functional_probe_retryable', 'functional_probe_passed',
        'leg1_queued', 'leg1_created'
      );
    SELECT COUNT(*) FILTER (
               WHERE work.created_at >= (
                   date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC'
               )
           ),
           COUNT(*) FILTER (WHERE work.created_at >= NOW() - INTERVAL '30 days')
    INTO v_day, v_30d
    FROM public.research_lab_source_add_work_items work
    JOIN public.research_lab_source_add_submission_current current
      ON current.submission_id = work.submission_id
    WHERE current.miner_hotkey = v_miner_hotkey
      AND work.work_kind = 'provenance'
      AND work.job_doc->>'admission_kind' = 'miner_submission';

    IF v_open >= p_max_open THEN RETURN jsonb_build_object('status', 'hotkey_open_cap'); END IF;
    IF v_day >= p_max_day THEN RETURN jsonb_build_object('status', 'hotkey_day_cap'); END IF;
    IF v_30d >= p_max_30d THEN RETURN jsonb_build_object('status', 'hotkey_30d_cap'); END IF;

    v_doc := p_record_doc || jsonb_build_object(
        'stage', 'provenance_queued',
        'stage_history', jsonb_build_array('submitted', 'manifest_validated', 'provenance_queued'),
        'source_identity_hash', p_identity_hash,
        'source_identity_version', 'v2'
    );
    INSERT INTO public.research_lab_source_add_submissions (
        submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
        precheck_status, precheck_doc, source_identity_hash, source_identity_version
    ) VALUES
        (v_submission_id, v_adapter_id, v_miner_hotkey, 'submitted', v_start_seq,
         p_record_doc || jsonb_build_object('stage', 'submitted'), '', '{}'::JSONB, p_identity_hash, 'v2'),
        (v_submission_id, v_adapter_id, v_miner_hotkey, 'manifest_validated', v_start_seq + 1,
         p_record_doc || jsonb_build_object('stage', 'manifest_validated'), '', '{}'::JSONB, p_identity_hash, 'v2'),
        (v_submission_id, v_adapter_id, v_miner_hotkey, 'provenance_queued', v_start_seq + 2,
         v_doc, '', '{}'::JSONB, p_identity_hash, 'v2');

    FOR v_identity IN
        SELECT DISTINCT ON (candidate.source_hash)
            candidate.source_hash,
            candidate.identity_version
        FROM (VALUES
            (p_identity_hash, 'v2'),
            (NULLIF(NULLIF(p_documentation_identity_hash, ''), p_identity_hash), 'v2'),
            (NULLIF(NULLIF(p_legacy_identity_hash, ''), p_identity_hash), 'v1')
        ) AS candidate(source_hash, identity_version)
        WHERE candidate.source_hash IS NOT NULL
        ORDER BY candidate.source_hash, candidate.identity_version DESC
    LOOP
        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
        FROM public.research_lab_source_add_identity_events
        WHERE identity_version = v_identity.identity_version
          AND source_identity_hash = v_identity.source_hash;
        INSERT INTO public.research_lab_source_add_identity_events (
            identity_version, source_identity_hash, submission_id, adapter_id,
            miner_hotkey, reservation_status, seq, reason
        ) VALUES (
            v_identity.identity_version,
            v_identity.source_hash, v_submission_id, v_adapter_id, v_miner_hotkey,
            'reserved', v_seq, 'atomic_admission'
        );
    END LOOP;

    INSERT INTO public.research_lab_source_add_work_items (
        work_id, submission_id, adapter_id, work_kind, work_status, priority, job_doc
    ) VALUES (
        p_work_id, v_submission_id, v_adapter_id, 'provenance', 'queued', 10,
        jsonb_build_object(
            'submission_id', v_submission_id,
            'identity_version', 'v2',
            'admission_kind', 'miner_submission',
            'terminal_resubmission', v_existing_terminal
        )
    );
    RETURN jsonb_build_object(
        'status', 'admitted', 'submission_id', v_submission_id,
        'adapter_id', v_adapter_id, 'stage', 'provenance_queued', 'work_id', p_work_id
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_begin_provider_execution(
    p_work_id TEXT,
    p_lease_token UUID
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
BEGIN
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'missing'); END IF;
    IF v_work.work_status <> 'leased'
       OR v_work.lease_token IS DISTINCT FROM p_lease_token THEN
        RETURN jsonb_build_object('status', 'lease_lost');
    END IF;
    IF v_work.work_kind NOT IN ('provenance', 'functional_probe', 'provisioning_smoke') THEN
        RAISE EXCEPTION 'SOURCE_ADD work kind has no provider execution';
    END IF;
    IF COALESCE(v_work.job_doc->>'provider_execution_recovery', '') <> '' THEN
        RETURN jsonb_build_object('status', 'recovery_only', 'work', to_jsonb(v_work));
    END IF;
    IF v_work.job_doc->>'provider_execution_state' = 'started'
       AND COALESCE((v_work.job_doc->>'provider_execution_attempt')::INTEGER, -1)
           = v_work.attempt_count THEN
        RETURN jsonb_build_object('status', 'already_started', 'work', to_jsonb(v_work));
    END IF;

    UPDATE public.research_lab_source_add_work_items
    SET job_doc = (job_doc - 'provider_execution_recovery') || jsonb_build_object(
            'provider_execution_state', 'started',
            'provider_execution_attempt', attempt_count,
            'provider_execution_started_at', NOW()
        ),
        updated_at = NOW()
    WHERE work_id = p_work_id
    RETURNING * INTO v_work;
    RETURN jsonb_build_object('status', 'started', 'work', to_jsonb(v_work));
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_claim_work(
    p_worker_id TEXT,
    p_lease_seconds INTEGER
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_row public.research_lab_source_add_work_items%ROWTYPE;
    v_token UUID := gen_random_uuid();
BEGIN
    IF p_worker_id = '' OR p_lease_seconds < 30 OR p_lease_seconds > 900 THEN
        RAISE EXCEPTION 'SOURCE_ADD work lease input is invalid';
    END IF;
    IF COALESCE((SELECT paused FROM public.research_lab_source_add_control WHERE singleton), TRUE) THEN
        RETURN jsonb_build_object('status', 'paused');
    END IF;
    SELECT w.* INTO v_row
    FROM public.research_lab_source_add_work_items w
    WHERE (
        (w.work_status IN ('queued', 'retry_wait') AND w.available_at <= NOW())
        OR (w.work_status = 'leased' AND w.lease_expires_at <= NOW())
    )
      AND (
        w.work_kind NOT IN ('functional_probe', 'provisioning_smoke')
        OR COALESCE(w.job_doc->>'host_hash', '') = ''
        OR NOT EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_work_items active
            WHERE active.work_id <> w.work_id
              AND active.work_status = 'leased'
              AND active.lease_expires_at > NOW()
              AND active.job_doc->>'host_hash' = w.job_doc->>'host_hash'
        )
      )
    ORDER BY w.priority ASC, w.available_at ASC, w.created_at ASC, w.work_id ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'empty'); END IF;

    IF v_row.work_kind IN ('functional_probe', 'provisioning_smoke')
       AND COALESCE(v_row.job_doc->>'host_hash', '') <> '' THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended(
                'source-add-host:' || (v_row.job_doc->>'host_hash'), 0
            )
        );
        IF EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_work_items active
            WHERE active.work_id <> v_row.work_id
              AND active.work_status = 'leased'
              AND active.lease_expires_at > NOW()
              AND active.job_doc->>'host_hash' = v_row.job_doc->>'host_hash'
        ) THEN
            RETURN jsonb_build_object('status', 'host_busy');
        END IF;
    END IF;

    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'leased',
        -- A crashed worker's expired lease reuses the same deterministic V2
        -- operation/attempt. Explicit retry_wait transitions advance attempts.
        attempt_count = CASE
            WHEN v_row.work_status = 'leased' THEN attempt_count
            ELSE LEAST(attempt_count + 1, 20)
        END,
        lease_token = v_token,
        leased_by = p_worker_id,
        lease_expires_at = NOW() + make_interval(secs => p_lease_seconds),
        job_doc = CASE
            WHEN v_row.work_status = 'leased'
                 AND v_row.job_doc->>'provider_execution_state' = 'started'
            THEN v_row.job_doc || jsonb_build_object(
                'provider_execution_recovery', 'uncertain_after_lease_expiry'
            )
            ELSE v_row.job_doc - 'provider_execution_recovery'
        END,
        updated_at = NOW()
    WHERE work_id = v_row.work_id
    RETURNING * INTO v_row;
    RETURN jsonb_build_object('status', 'claimed', 'work', to_jsonb(v_row));
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finish_work(
    p_work_id TEXT,
    p_lease_token UUID,
    p_disposition TEXT,
    p_stage TEXT,
    p_submission_doc JSONB,
    p_precheck_status TEXT,
    p_precheck_doc JSONB,
    p_result_doc JSONB,
    p_functional_attempt JSONB,
    p_probe_config JSONB,
    p_next_work JSONB,
    p_reward_intent JSONB,
    p_available_at TIMESTAMPTZ,
    p_release_identity BOOLEAN
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_work public.research_lab_source_add_work_items%ROWTYPE;
    v_seq INTEGER;
    v_identity RECORD;
    v_attempt RECORD;
    v_config RECORD;
    v_intent RECORD;
    v_next RECORD;
    v_stage TEXT;
    v_transition_stage TEXT;
    v_transition_stages TEXT[] := ARRAY[]::TEXT[];
    v_evaluation_mode TEXT;
BEGIN
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id
    FOR UPDATE;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'missing'); END IF;
    IF v_work.work_status = 'completed' THEN RETURN jsonb_build_object('status', 'already_completed'); END IF;
    IF v_work.work_status <> 'leased' OR v_work.lease_token IS DISTINCT FROM p_lease_token THEN
        RETURN jsonb_build_object('status', 'lease_lost');
    END IF;
    IF p_disposition NOT IN ('complete', 'retry') THEN RAISE EXCEPTION 'invalid SOURCE_ADD disposition'; END IF;
    IF NULLIF(p_stage, '') IS NOT NULL AND p_stage NOT IN (
        'provenance_precheck_passed', 'needs_manual_review', 'rejected_precheck',
        'functional_probe_queued', 'awaiting_operator_credential',
        'functional_probe_retryable', 'functional_probe_passed',
        'functional_probe_failed', 'leg1_queued', 'leg1_created'
    ) THEN
        RAISE EXCEPTION 'invalid SOURCE_ADD transition stage';
    END IF;
    IF p_disposition = 'retry' AND p_available_at IS NULL THEN
        RAISE EXCEPTION 'SOURCE_ADD retry requires available_at';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-submission:' || v_work.submission_id, 0)
    );
    v_stage := NULLIF(p_stage, '');
    -- Keep every externally meaningful state transition in append-only history.
    -- A single worker transaction may both prove provenance and queue a probe,
    -- or prove a functional response and queue Leg 1.
    IF v_work.work_kind = 'provenance'
       AND p_precheck_status = 'provenance_precheck_passed'
       AND v_stage IS DISTINCT FROM 'provenance_precheck_passed' THEN
        v_transition_stages := array_append(
            v_transition_stages, 'provenance_precheck_passed'
        );
    END IF;
    IF v_work.work_kind = 'functional_probe'
       AND COALESCE(p_functional_attempt->>'result_status', '') = 'passed'
       AND v_stage IS DISTINCT FROM 'functional_probe_passed' THEN
        v_transition_stages := array_append(
            v_transition_stages, 'functional_probe_passed'
        );
    END IF;
    IF v_stage IS NOT NULL THEN
        v_transition_stages := array_append(v_transition_stages, v_stage);
    END IF;
    FOREACH v_transition_stage IN ARRAY v_transition_stages
    LOOP
        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
        FROM public.research_lab_source_add_submissions
        WHERE submission_id = v_work.submission_id;
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
            precheck_status, precheck_doc, source_identity_hash, source_identity_version
        )
        SELECT
            v_work.submission_id,
            v_work.adapter_id,
            current.miner_hotkey,
            v_transition_stage,
            v_seq,
            p_submission_doc || jsonb_build_object('stage', v_transition_stage),
            p_precheck_status,
            p_precheck_doc,
            current.source_identity_hash,
            'v2'
        FROM public.research_lab_source_add_submission_current current
        WHERE current.submission_id = v_work.submission_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'SOURCE_ADD current submission disappeared';
        END IF;
    END LOOP;

    IF p_functional_attempt <> '{}'::JSONB THEN
        v_evaluation_mode := CASE v_work.work_kind
            WHEN 'functional_probe' THEN 'functional_probe'
            WHEN 'provisioning_smoke' THEN 'provisioning_smoke'
            ELSE ''
        END;
        IF p_functional_attempt->>'attempt_ref' !~ '^source_add_probe_attempt:[0-9a-f]{16}$'
           OR p_functional_attempt->>'config_ref' !~ '^source_add_probe_config:[0-9a-f]{16}$'
           OR v_evaluation_mode = ''
           OR COALESCE(p_functional_attempt->>'evaluation_mode', '') <> v_evaluation_mode
           OR COALESCE(p_functional_attempt->'result_doc'->>'submission_id', '') <> v_work.submission_id
           OR COALESCE(p_functional_attempt->'result_doc'->>'adapter_id', '') <> v_work.adapter_id
           OR COALESCE(p_functional_attempt->'result_doc'->>'config_ref', '') <> p_functional_attempt->>'config_ref'
           OR COALESCE(p_functional_attempt->'result_doc'->>'evaluation_mode', '') <> v_evaluation_mode
           OR COALESCE(p_functional_attempt->'result_doc'->>'result_status', '') <> p_functional_attempt->>'result_status'
           OR COALESCE(p_functional_attempt->'result_doc'->>'route_hash', '') <> COALESCE(p_functional_attempt->>'route_hash', '') THEN
            RAISE EXCEPTION 'SOURCE_ADD functional attempt binding is invalid';
        END IF;
        INSERT INTO public.research_lab_source_add_functional_probe_attempts (
            attempt_ref, submission_id, adapter_id, work_id, attempt_number,
            evaluation_mode, config_ref, result_status, route_hash, response_hash,
            status_class, content_type, byte_count, duration_ms, retry_after_seconds,
            reason_codes, receipt_hash,
            business_artifact_hash, result_doc
        ) VALUES (
            p_functional_attempt->>'attempt_ref', v_work.submission_id, v_work.adapter_id,
            p_work_id, v_work.attempt_count, v_evaluation_mode,
            p_functional_attempt->>'config_ref', p_functional_attempt->>'result_status',
            COALESCE(p_functional_attempt->>'route_hash', ''),
            COALESCE(p_functional_attempt->>'response_hash', ''),
            COALESCE(p_functional_attempt->>'status_class', ''),
            COALESCE(p_functional_attempt->>'content_type', ''),
            COALESCE((p_functional_attempt->>'byte_count')::INTEGER, 0),
            COALESCE((p_functional_attempt->>'duration_ms')::INTEGER, 0),
            COALESCE((p_functional_attempt->>'retry_after_seconds')::INTEGER, 0),
            COALESCE(p_functional_attempt->'reason_codes', '[]'::JSONB),
            COALESCE(p_functional_attempt->>'receipt_hash', ''),
            COALESCE(p_functional_attempt->>'business_artifact_hash', ''),
            COALESCE(p_functional_attempt->'result_doc', '{}'::JSONB)
        ) ON CONFLICT (work_id, attempt_number) DO NOTHING;
        SELECT * INTO v_attempt
        FROM public.research_lab_source_add_functional_probe_attempts
        WHERE work_id = p_work_id AND attempt_number = v_work.attempt_count;
        IF NOT FOUND
           OR v_attempt.attempt_ref <> p_functional_attempt->>'attempt_ref'
           OR v_attempt.result_doc <> p_functional_attempt->'result_doc'
           OR v_attempt.receipt_hash <> p_functional_attempt->>'receipt_hash'
           OR v_attempt.business_artifact_hash <> p_functional_attempt->>'business_artifact_hash' THEN
            RAISE EXCEPTION 'SOURCE_ADD functional attempt idempotency differs';
        END IF;
    END IF;

    IF p_probe_config <> '{}'::JSONB THEN
        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
        FROM public.research_lab_source_add_probe_config_events
        WHERE submission_id = v_work.submission_id;
        INSERT INTO public.research_lab_source_add_probe_config_events (
            config_ref, submission_id, adapter_id, config_status, seq,
            probe_doc, credential_envelope, actor_ref
        ) VALUES (
            p_probe_config->>'config_ref', v_work.submission_id, v_work.adapter_id,
            'active', v_seq, p_probe_config->'probe_doc',
            COALESCE(p_probe_config->'credential_envelope', '{}'::JSONB),
            COALESCE(p_probe_config->>'actor_ref', 'system:auto-probe')
        ) ON CONFLICT (config_ref) DO NOTHING;
        SELECT * INTO v_config
        FROM public.research_lab_source_add_probe_config_events
        WHERE config_ref = p_probe_config->>'config_ref';
        IF NOT FOUND OR v_config.submission_id <> v_work.submission_id
           OR v_config.probe_doc <> p_probe_config->'probe_doc'
           OR v_config.credential_envelope <> COALESCE(p_probe_config->'credential_envelope', '{}'::JSONB) THEN
            RAISE EXCEPTION 'SOURCE_ADD probe config idempotency differs';
        END IF;
    END IF;

    IF p_reward_intent <> '{}'::JSONB THEN
        INSERT INTO public.research_lab_source_add_reward_intents (
            intent_id, submission_id, adapter_id, miner_hotkey, intent_status,
            functional_receipt_hash, business_artifact_hash
        ) VALUES (
            p_reward_intent->>'intent_id', v_work.submission_id, v_work.adapter_id,
            p_reward_intent->>'miner_hotkey', 'queued',
            p_reward_intent->>'functional_receipt_hash',
            p_reward_intent->>'business_artifact_hash'
        ) ON CONFLICT (adapter_id, leg) DO NOTHING;
        SELECT * INTO v_intent
        FROM public.research_lab_source_add_reward_intents
        WHERE adapter_id = v_work.adapter_id AND leg = 1;
        IF NOT FOUND OR v_intent.submission_id <> v_work.submission_id
           OR v_intent.functional_receipt_hash <> p_reward_intent->>'functional_receipt_hash'
           OR v_intent.business_artifact_hash <> p_reward_intent->>'business_artifact_hash' THEN
            RAISE EXCEPTION 'SOURCE_ADD reward intent idempotency differs';
        END IF;
    END IF;

    IF p_next_work <> '{}'::JSONB THEN
        INSERT INTO public.research_lab_source_add_work_items (
            work_id, submission_id, adapter_id, work_kind, work_status,
            priority, available_at, job_doc
        ) VALUES (
            p_next_work->>'work_id', v_work.submission_id, v_work.adapter_id,
            p_next_work->>'work_kind', 'queued',
            COALESCE((p_next_work->>'priority')::INTEGER, 100),
            COALESCE((p_next_work->>'available_at')::TIMESTAMPTZ, NOW()),
            COALESCE(p_next_work->'job_doc', '{}'::JSONB)
        ) ON CONFLICT (work_id) DO NOTHING;
        SELECT * INTO v_next FROM public.research_lab_source_add_work_items
        WHERE work_id = p_next_work->>'work_id';
        IF NOT FOUND OR v_next.submission_id <> v_work.submission_id
           OR v_next.work_kind <> p_next_work->>'work_kind'
           OR v_next.job_doc <> COALESCE(p_next_work->'job_doc', '{}'::JSONB) THEN
            RAISE EXCEPTION 'SOURCE_ADD next work idempotency differs';
        END IF;
    END IF;

    IF p_release_identity THEN
        FOR v_identity IN
            SELECT * FROM public.research_lab_source_add_identity_current
            WHERE submission_id = v_work.submission_id AND reservation_status = 'reserved'
        LOOP
            INSERT INTO public.research_lab_source_add_identity_events (
                identity_version, source_identity_hash, submission_id, adapter_id,
                miner_hotkey, reservation_status, seq, reason
            ) VALUES (
                v_identity.identity_version, v_identity.source_identity_hash,
                v_identity.submission_id, v_identity.adapter_id, v_identity.miner_hotkey,
                'released', v_identity.seq + 1, 'terminal_rejection'
            );
        END LOOP;
    END IF;

    IF p_disposition = 'retry' THEN
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'retry_wait', available_at = p_available_at,
            lease_token = NULL, leased_by = '', lease_expires_at = NULL,
            job_doc = job_doc
                - 'provider_execution_state'
                - 'provider_execution_attempt'
                - 'provider_execution_started_at'
                - 'provider_execution_recovery',
            result_doc = p_result_doc, updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN jsonb_build_object('status', 'retry_wait', 'available_at', p_available_at);
    END IF;

    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'completed', lease_token = NULL, leased_by = '',
        lease_expires_at = NULL,
        job_doc = job_doc
            - 'provider_execution_state'
            - 'provider_execution_attempt'
            - 'provider_execution_started_at'
            - 'provider_execution_recovery',
        result_doc = p_result_doc,
        completed_at = NOW(), updated_at = NOW()
    WHERE work_id = p_work_id;
    RETURN jsonb_build_object('status', 'completed');
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_configure_probe(
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
DECLARE
    v_current RECORD;
    v_existing RECORD;
    v_seq INTEGER;
    v_config_exists BOOLEAN := FALSE;
BEGIN
    IF p_config_ref !~ '^source_add_probe_config:[0-9a-f]{16}$'
       OR p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR p_host_hash !~ '^sha256:[0-9a-f]{64}$'
       OR jsonb_typeof(p_probe_doc) <> 'object'
       OR jsonb_typeof(p_credential_envelope) <> 'object'
       OR btrim(p_actor_ref) = ''
       OR p_probe_doc::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])' THEN
        RAISE EXCEPTION 'SOURCE_ADD probe configuration input is invalid';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-submission:' || p_submission_id, 0)
    );
    SELECT * INTO v_current FROM public.research_lab_source_add_submission_current
    WHERE submission_id = p_submission_id;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'missing'); END IF;
    SELECT * INTO v_existing
    FROM public.research_lab_source_add_probe_config_events
    WHERE config_ref = p_config_ref;
    v_config_exists := FOUND;
    IF FOUND THEN
        IF v_existing.submission_id <> p_submission_id
           OR v_existing.probe_doc <> p_probe_doc
           OR v_existing.credential_envelope <> p_credential_envelope THEN
            RAISE EXCEPTION 'SOURCE_ADD config reference collision';
        END IF;
    END IF;
    SELECT * INTO v_existing FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id;
    IF FOUND THEN
        IF v_existing.submission_id <> p_submission_id
           OR v_existing.work_kind <> 'functional_probe'
           OR (v_existing.job_doc
               - 'provider_execution_state'
               - 'provider_execution_attempt'
               - 'provider_execution_started_at'
               - 'provider_execution_recovery') <> jsonb_build_object(
               'config_ref', p_config_ref, 'host_hash', p_host_hash
           ) THEN
            RAISE EXCEPTION 'SOURCE_ADD functional work idempotency differs';
        END IF;
        RETURN jsonb_build_object(
            'status', 'already_configured', 'work_id', p_work_id,
            'config_ref', p_config_ref, 'work_status', v_existing.work_status,
            'stage', v_current.stage
        );
    END IF;
    IF v_current.stage IN ('rejected', 'rejected_precheck', 'functional_probe_failed') THEN
        RETURN jsonb_build_object('status', 'terminal');
    END IF;
    IF v_current.precheck_status <> 'provenance_precheck_passed' THEN
        RETURN jsonb_build_object('status', 'provenance_required');
    END IF;
    IF NOT v_config_exists THEN
        SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
        FROM public.research_lab_source_add_probe_config_events
        WHERE submission_id = p_submission_id;
        INSERT INTO public.research_lab_source_add_probe_config_events (
            config_ref, submission_id, adapter_id, config_status, seq,
            probe_doc, credential_envelope, actor_ref
        ) VALUES (
            p_config_ref, p_submission_id, v_current.adapter_id, 'active', v_seq,
            p_probe_doc, p_credential_envelope, left(p_actor_ref, 200)
        );
    END IF;
    INSERT INTO public.research_lab_source_add_work_items (
        work_id, submission_id, adapter_id, work_kind, work_status, priority, job_doc
    ) VALUES (
        p_work_id, p_submission_id, v_current.adapter_id, 'functional_probe',
        'queued', 20, jsonb_build_object('config_ref', p_config_ref, 'host_hash', p_host_hash)
    );
    SELECT * INTO v_existing FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id;
    IF NOT FOUND OR v_existing.submission_id <> p_submission_id
       OR v_existing.work_kind <> 'functional_probe'
       OR (v_existing.job_doc
           - 'provider_execution_state'
           - 'provider_execution_attempt'
           - 'provider_execution_started_at'
           - 'provider_execution_recovery')
          <> jsonb_build_object('config_ref', p_config_ref, 'host_hash', p_host_hash) THEN
        RAISE EXCEPTION 'SOURCE_ADD functional work idempotency differs';
    END IF;
    SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
    FROM public.research_lab_source_add_submissions WHERE submission_id = p_submission_id;
    IF v_current.stage <> 'functional_probe_queued' THEN
      INSERT INTO public.research_lab_source_add_submissions (
        submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
        precheck_status, precheck_doc, source_identity_hash, source_identity_version
      ) VALUES (
        p_submission_id, v_current.adapter_id, v_current.miner_hotkey,
        'functional_probe_queued', v_seq,
        v_current.submission_doc || jsonb_build_object('stage', 'functional_probe_queued'),
        v_current.precheck_status, v_current.precheck_doc,
        v_current.source_identity_hash, 'v2'
      );
    END IF;
    RETURN jsonb_build_object('status', 'queued', 'work_id', p_work_id, 'config_ref', p_config_ref);
END;
$$;

DROP FUNCTION IF EXISTS public.research_lab_source_add_requeue_provenance(
    TEXT, TEXT, TEXT, TEXT, TEXT
);

CREATE OR REPLACE FUNCTION public.research_lab_source_add_requeue_provenance(
    p_submission_id TEXT,
    p_identity_hash TEXT,
    p_documentation_identity_hash TEXT,
    p_legacy_identity_hash TEXT,
    p_work_id TEXT,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    v_current RECORD;
    v_hash TEXT;
    v_version TEXT;
    v_seq INTEGER;
    v_identity RECORD;
BEGIN
    IF p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR p_identity_hash !~ '^sha256:[0-9a-f]{64}$'
       OR (COALESCE(p_documentation_identity_hash, '') <> ''
           AND p_documentation_identity_hash !~ '^sha256:[0-9a-f]{64}$')
       OR (COALESCE(p_legacy_identity_hash, '') <> ''
           AND p_legacy_identity_hash !~ '^sha256:[0-9a-f]{64}$')
       OR p_work_id !~ '^source_add_work:[0-9a-f]{16}$'
       OR btrim(p_actor_ref) = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD provenance requeue input is invalid';
    END IF;
    FOR v_hash IN
        SELECT DISTINCT item FROM unnest(ARRAY[
            p_identity_hash,
            NULLIF(NULLIF(p_documentation_identity_hash, ''), p_identity_hash),
            NULLIF(NULLIF(p_legacy_identity_hash, ''), p_identity_hash)
        ]) item
        WHERE item IS NOT NULL ORDER BY item
    LOOP
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended('source-add-identity:' || v_hash, 0)
        );
    END LOOP;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-submission:' || p_submission_id, 0)
    );
    SELECT * INTO v_current
    FROM public.research_lab_source_add_submission_current
    WHERE submission_id = p_submission_id;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'missing'); END IF;
    IF v_current.stage IN ('accepted', 'rejected', 'rejected_precheck', 'functional_probe_failed') THEN
        RETURN jsonb_build_object('status', 'terminal');
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.research_lab_source_add_identity_current identity
        WHERE identity.reservation_status = 'reserved'
          AND identity.submission_id <> p_submission_id
          AND identity.source_identity_hash IN (
              p_identity_hash,
              NULLIF(p_documentation_identity_hash, ''),
              NULLIF(p_legacy_identity_hash, '')
          )
    ) OR EXISTS (
        SELECT 1 FROM public.research_lab_source_catalog catalog
        WHERE catalog.adapter_id <> v_current.adapter_id
          AND catalog.source_identity_hash IN (
              p_identity_hash,
              NULLIF(p_documentation_identity_hash, ''),
              NULLIF(p_legacy_identity_hash, '')
          )
    ) THEN
        RETURN jsonb_build_object('status', 'duplicate');
    END IF;

    FOR v_identity IN
        SELECT DISTINCT ON (candidate.source_hash)
            candidate.source_hash,
            candidate.identity_version
        FROM (VALUES
            (p_identity_hash, 'v2'),
            (NULLIF(NULLIF(p_documentation_identity_hash, ''), p_identity_hash), 'v2'),
            (NULLIF(NULLIF(p_legacy_identity_hash, ''), p_identity_hash), 'v1')
        ) AS candidate(source_hash, identity_version)
        WHERE candidate.source_hash IS NOT NULL
        ORDER BY candidate.source_hash, candidate.identity_version DESC
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM public.research_lab_source_add_identity_current current_identity
            WHERE current_identity.identity_version = v_identity.identity_version
              AND current_identity.source_identity_hash = v_identity.source_hash
              AND current_identity.submission_id = p_submission_id
              AND current_identity.reservation_status = 'reserved'
        ) THEN
            SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
            FROM public.research_lab_source_add_identity_events
            WHERE identity_version = v_identity.identity_version
              AND source_identity_hash = v_identity.source_hash;
            INSERT INTO public.research_lab_source_add_identity_events (
                identity_version, source_identity_hash, submission_id, adapter_id,
                miner_hotkey, reservation_status, seq, reason
            ) VALUES (
                v_identity.identity_version, v_identity.source_hash, p_submission_id,
                v_current.adapter_id, v_current.miner_hotkey, 'reserved', v_seq,
                'operator_provenance_recheck'
            );
        END IF;
    END LOOP;

    INSERT INTO public.research_lab_source_add_work_items (
        work_id, submission_id, adapter_id, work_kind, work_status, priority, job_doc
    ) VALUES (
        p_work_id, p_submission_id, v_current.adapter_id, 'provenance', 'queued', 10,
        jsonb_build_object('submission_id', p_submission_id, 'actor_ref', left(p_actor_ref, 200))
    ) ON CONFLICT (work_id) DO NOTHING;
    SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
    FROM public.research_lab_source_add_submissions WHERE submission_id = p_submission_id;
    INSERT INTO public.research_lab_source_add_submissions (
        submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
        precheck_status, precheck_doc, source_identity_hash, source_identity_version
    ) VALUES (
        p_submission_id, v_current.adapter_id, v_current.miner_hotkey,
        'provenance_queued', v_seq,
        v_current.submission_doc || jsonb_build_object(
            'stage', 'provenance_queued',
            'source_identity_hash', p_identity_hash,
            'source_identity_version', 'v2'
        ),
        v_current.precheck_status, v_current.precheck_doc, p_identity_hash, 'v2'
    );
    RETURN jsonb_build_object(
        'status', 'queued', 'work_id', p_work_id, 'stage', 'provenance_queued'
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_set_paused(
    p_paused BOOLEAN,
    p_reason TEXT,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF btrim(p_reason) = '' OR btrim(p_actor_ref) = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD pause reason and actor are required';
    END IF;
    UPDATE public.research_lab_source_add_control
    SET paused = p_paused, reason = left(p_reason, 500),
        actor_ref = left(p_actor_ref, 200), updated_at = NOW()
    WHERE singleton;
    RETURN (SELECT to_jsonb(c) FROM public.research_lab_source_add_control c WHERE singleton);
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_reserve_leg1_slot(
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
    v_day DATE := (NOW() AT TIME ZONE 'UTC')::DATE;
    v_created INTEGER;
    v_reserved INTEGER;
    v_slot INTEGER;
    v_token UUID := gen_random_uuid();
    v_existing TEXT;
    v_existing_slot public.research_lab_source_add_reward_slots%ROWTYPE;
BEGIN
    IF p_daily_cap < 1 OR p_daily_cap > 100 OR p_slot_lease_seconds < 30 OR p_slot_lease_seconds > 1800 THEN
        RAISE EXCEPTION 'SOURCE_ADD reward slot policy is invalid';
    END IF;
    SELECT * INTO v_work FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id FOR UPDATE;
    IF NOT FOUND OR v_work.work_status <> 'leased' OR v_work.lease_token IS DISTINCT FROM p_work_lease_token THEN
        RETURN jsonb_build_object('status', 'lease_lost');
    END IF;
    SELECT * INTO v_intent FROM public.research_lab_source_add_reward_intents
    WHERE intent_id = p_intent_id FOR UPDATE;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'intent_missing'); END IF;

    SELECT reward_ref INTO v_existing
    FROM public.research_lab_source_add_reward_obligations
    WHERE adapter_id = v_intent.adapter_id AND leg = 1;
    IF FOUND THEN
        UPDATE public.research_lab_source_add_reward_slots
        SET slot_status = 'released', updated_at = NOW()
        WHERE intent_id = p_intent_id AND slot_status = 'reserved';
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'finalized', reward_ref = v_existing, updated_at = NOW()
        WHERE intent_id = p_intent_id;
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'completed', result_doc = jsonb_build_object('status', 'already_created', 'reward_ref', v_existing),
            completed_at = NOW(), lease_token = NULL, leased_by = '', lease_expires_at = NULL, updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN jsonb_build_object('status', 'already_created', 'reward_ref', v_existing);
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-leg1-day:' || v_day::TEXT, 0)
    );
    UPDATE public.research_lab_source_add_reward_slots
    SET slot_status = 'expired', updated_at = NOW()
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
        UPDATE public.research_lab_source_add_reward_slots
        SET work_id = p_work_id, lease_token = v_token,
            lease_expires_at = NOW() + make_interval(secs => p_slot_lease_seconds),
            updated_at = NOW()
        WHERE slot_id = v_existing_slot.slot_id;
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'leased', updated_at = NOW() WHERE intent_id = p_intent_id;
        RETURN jsonb_build_object(
            'status', 'reserved', 'slot_day', v_day,
            'slot_number', v_existing_slot.slot_number,
            'slot_lease_token', v_token,
            'lease_expires_at', NOW() + make_interval(secs => p_slot_lease_seconds)
        );
    END IF;

    SELECT COUNT(*) INTO v_created
    FROM public.research_lab_source_add_reward_events
    WHERE reason IN ('leg1_provenance_precheck_passed', 'leg1_functional_probe_passed')
      AND created_at >= (v_day::TIMESTAMP AT TIME ZONE 'UTC')
      AND created_at < ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC');
    SELECT COUNT(*) INTO v_reserved
    FROM public.research_lab_source_add_reward_slots
    WHERE slot_day = v_day AND slot_status = 'reserved' AND lease_expires_at > NOW();
    IF v_created + v_reserved >= p_daily_cap THEN
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'retry_wait',
            available_at = ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
            updated_at = NOW()
        WHERE intent_id = p_intent_id;
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'retry_wait',
            available_at = ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
            lease_token = NULL, leased_by = '', lease_expires_at = NULL,
            result_doc = jsonb_build_object('status', 'daily_cap_fifo'), updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN jsonb_build_object(
            'status', 'daily_cap_fifo',
            'available_at', ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC')
        );
    END IF;

    SELECT number INTO v_slot
    FROM generate_series(1, p_daily_cap) number
    WHERE NOT EXISTS (
        SELECT 1 FROM public.research_lab_source_add_reward_slots s
        WHERE s.slot_day = v_day AND s.slot_number = number
          AND s.slot_status IN ('reserved', 'finalized')
    )
    ORDER BY number LIMIT 1;
    IF v_slot IS NULL THEN RAISE EXCEPTION 'SOURCE_ADD reward slot accounting differs'; END IF;

    INSERT INTO public.research_lab_source_add_reward_slots (
        slot_day, slot_number, intent_id, work_id, slot_status,
        lease_token, lease_expires_at
    ) VALUES (
        v_day, v_slot, p_intent_id, p_work_id, 'reserved', v_token,
        NOW() + make_interval(secs => p_slot_lease_seconds)
    );
    UPDATE public.research_lab_source_add_reward_intents
    SET intent_status = 'leased', updated_at = NOW() WHERE intent_id = p_intent_id;
    RETURN jsonb_build_object(
        'status', 'reserved', 'slot_day', v_day, 'slot_number', v_slot,
        'slot_lease_token', v_token, 'lease_expires_at', NOW() + make_interval(secs => p_slot_lease_seconds)
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_leg1(
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
    v_day DATE;
    v_created INTEGER;
    v_existing TEXT;
    v_seq INTEGER;
    v_probe public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_current RECORD;
BEGIN
    IF p_daily_cap < 1 OR p_daily_cap > 100
       OR p_reward->>'reward_ref' !~ '^source_add_reward:[0-9a-f]{16}$'
       OR COALESCE(p_reward->>'reward_kind', '') <> 'source_acceptance'
       OR COALESCE((p_reward->>'alpha_percent')::NUMERIC, 0) <= 0
       OR COALESCE((p_reward->>'reward_epochs')::INTEGER, 0) <= 0
       OR COALESCE((p_reward->>'start_epoch')::INTEGER, -1) < 0
       OR p_reward->>'decision_receipt_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR p_reward->>'decision_artifact_hash' !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(p_reward->'trigger_evidence_doc'->>'functional_probe_passed', '') <> 'true' THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 reward input is invalid';
    END IF;
    SELECT * INTO v_work FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id FOR UPDATE;
    IF NOT FOUND OR v_work.work_status <> 'leased' OR v_work.work_kind <> 'leg1_reward'
       OR v_work.lease_token IS DISTINCT FROM p_work_lease_token THEN
        RETURN jsonb_build_object('status', 'lease_lost');
    END IF;
    SELECT * INTO v_intent FROM public.research_lab_source_add_reward_intents
    WHERE intent_id = p_intent_id FOR UPDATE;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'intent_missing'); END IF;
    IF v_work.submission_id <> v_intent.submission_id
       OR v_work.adapter_id <> v_intent.adapter_id
       OR v_intent.intent_status <> 'leased' THEN
        RAISE EXCEPTION 'SOURCE_ADD reward intent scope differs';
    END IF;
    -- An intent can have expired/released audit rows from an earlier lease or
    -- UTC day. Bind finalization to the exact currently reserved slot token.
    SELECT * INTO v_slot FROM public.research_lab_source_add_reward_slots
    WHERE intent_id = p_intent_id
      AND slot_status = 'reserved'
      AND lease_token = p_slot_lease_token
    FOR UPDATE;
    IF NOT FOUND OR v_slot.slot_status <> 'reserved'
       OR v_slot.work_id <> p_work_id
       OR v_slot.lease_token IS DISTINCT FROM p_slot_lease_token
       OR v_slot.lease_expires_at <= NOW() THEN
        RETURN jsonb_build_object('status', 'slot_lost');
    END IF;
    v_day := v_slot.slot_day;
    IF v_day <> (NOW() AT TIME ZONE 'UTC')::DATE THEN
        UPDATE public.research_lab_source_add_reward_slots
        SET slot_status = 'released', updated_at = NOW() WHERE slot_id = v_slot.slot_id;
        UPDATE public.research_lab_source_add_reward_intents
        SET intent_status = 'retry_wait', available_at = NOW(), updated_at = NOW()
        WHERE intent_id = p_intent_id;
        UPDATE public.research_lab_source_add_work_items
        SET work_status = 'retry_wait', available_at = NOW(), lease_token = NULL,
            leased_by = '', lease_expires_at = NULL, updated_at = NOW()
        WHERE work_id = p_work_id;
        RETURN jsonb_build_object('status', 'slot_day_rolled');
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-leg1-day:' || v_day::TEXT, 0)
    );
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-leg1-adapter:' || v_intent.adapter_id, 0)
    );

    SELECT * INTO v_probe
    FROM public.research_lab_source_add_functional_probe_current
    WHERE submission_id = v_intent.submission_id;
    IF NOT FOUND OR v_probe.adapter_id <> v_intent.adapter_id
       OR v_probe.result_status <> 'passed'
       OR v_probe.receipt_hash <> v_intent.functional_receipt_hash
       OR v_probe.business_artifact_hash <> v_intent.business_artifact_hash
       OR p_reward->'trigger_evidence_doc'->>'attempt_ref' <> v_probe.attempt_ref
       OR p_reward->'trigger_evidence_doc'->>'functional_probe_receipt_hash' <> v_probe.receipt_hash
       OR p_reward->'trigger_evidence_doc'->>'business_artifact_hash' <> v_probe.business_artifact_hash
       OR p_reward->'trigger_evidence_doc'->>'functional_probe_result_hash' <> v_probe.business_artifact_hash
       OR p_reward->'trigger_evidence_doc'->>'evaluator_version' <> v_probe.result_doc->>'evaluator_version'
       OR p_reward->'trigger_evidence_doc'->>'route_hash' <> v_probe.route_hash THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 functional proof differs';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM public.research_lab_attested_execution_receipts_v2 receipt
        JOIN public.research_lab_attested_business_artifact_links_v2 link
          ON link.receipt_hash = receipt.receipt_hash
        WHERE receipt.receipt_hash = v_probe.receipt_hash
          AND receipt.role = 'gateway_coordinator'
          AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
          AND receipt.receipt_status = 'succeeded'
          AND receipt.output_root = v_probe.business_artifact_hash
          AND link.artifact_kind = 'source_add_functional_probe'
          AND link.artifact_ref = v_probe.attempt_ref
          AND link.artifact_hash = v_probe.business_artifact_hash
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 measured receipt is unavailable';
    END IF;
    IF NOT EXISTS (
        SELECT 1
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
          AND link.artifact_hash = p_reward->>'decision_artifact_hash'
    ) THEN
        RAISE EXCEPTION 'SOURCE_ADD Leg 1 reward decision receipt is unavailable';
    END IF;

    SELECT reward_ref INTO v_existing FROM public.research_lab_source_add_reward_obligations
    WHERE adapter_id = v_intent.adapter_id AND leg = 1;
    IF NOT FOUND THEN
        SELECT COUNT(*) INTO v_created
        FROM public.research_lab_source_add_reward_events
        WHERE reason IN ('leg1_provenance_precheck_passed', 'leg1_functional_probe_passed')
          AND created_at >= (v_day::TIMESTAMP AT TIME ZONE 'UTC')
          AND created_at < ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC');
        IF v_created >= p_daily_cap THEN
            UPDATE public.research_lab_source_add_reward_slots SET slot_status = 'released', updated_at = NOW()
            WHERE slot_id = v_slot.slot_id;
            UPDATE public.research_lab_source_add_reward_intents
            SET intent_status = 'retry_wait',
                available_at = ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
                updated_at = NOW()
            WHERE intent_id = p_intent_id;
            UPDATE public.research_lab_source_add_work_items
            SET work_status = 'retry_wait',
                available_at = ((v_day + 1)::TIMESTAMP AT TIME ZONE 'UTC'),
                lease_token = NULL, leased_by = '', lease_expires_at = NULL, updated_at = NOW()
            WHERE work_id = p_work_id;
            RETURN jsonb_build_object('status', 'daily_cap_fifo');
        END IF;
        INSERT INTO public.research_lab_source_add_reward_obligations (
            reward_ref, adapter_id, catalog_id, miner_hotkey, leg, reward_kind,
            alpha_percent, reward_epochs, start_epoch, trigger_evidence_doc, public_label
        ) VALUES (
            p_reward->>'reward_ref', v_intent.adapter_id, NULL, v_intent.miner_hotkey,
            1, p_reward->>'reward_kind', (p_reward->>'alpha_percent')::NUMERIC,
            (p_reward->>'reward_epochs')::INTEGER, (p_reward->>'start_epoch')::INTEGER,
            p_reward->'trigger_evidence_doc', COALESCE(p_reward->>'public_label', '')
        );
        INSERT INTO public.research_lab_source_add_reward_events (
            reward_ref, seq, reward_status, reason
        ) VALUES (p_reward->>'reward_ref', 0, p_reward->>'state', 'leg1_functional_probe_passed');
        v_existing := p_reward->>'reward_ref';
    END IF;

    UPDATE public.research_lab_source_add_reward_slots
    SET slot_status = 'finalized', reward_ref = v_existing, updated_at = NOW()
    WHERE slot_id = v_slot.slot_id;
    UPDATE public.research_lab_source_add_reward_intents
    SET intent_status = 'finalized', reward_ref = v_existing, updated_at = NOW()
    WHERE intent_id = p_intent_id;
    UPDATE public.research_lab_source_add_work_items
    SET work_status = 'completed', result_doc = jsonb_build_object('status', 'created', 'reward_ref', v_existing),
        completed_at = NOW(), lease_token = NULL, leased_by = '', lease_expires_at = NULL, updated_at = NOW()
    WHERE work_id = p_work_id;

    SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
    FROM public.research_lab_source_add_submissions WHERE submission_id = v_intent.submission_id;
    INSERT INTO public.research_lab_source_add_submissions (
        submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
        precheck_status, precheck_doc, source_identity_hash, source_identity_version
    )
    SELECT
        v_intent.submission_id, v_intent.adapter_id, v_intent.miner_hotkey,
        'leg1_created', v_seq,
        current.submission_doc || jsonb_build_object('stage', 'leg1_created', 'leg1_reward_ref', v_existing),
        current.precheck_status, current.precheck_doc, current.source_identity_hash, 'v2'
    FROM public.research_lab_source_add_submission_current current
    WHERE current.submission_id = v_intent.submission_id;
    RETURN jsonb_build_object('status', 'created', 'reward_ref', v_existing);
END;
$$;

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
    SELECT * INTO v_work
    FROM public.research_lab_source_add_work_items
    WHERE work_id = p_work_id;
    IF FOUND THEN
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

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_provision(
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
    v_current RECORD;
    v_probe public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_config RECORD;
    v_smoke public.research_lab_source_add_functional_probe_attempts%ROWTYPE;
    v_catalog public.research_lab_source_catalog%ROWTYPE;
    v_catalog_id TEXT;
    v_adapter_id TEXT := p_catalog_row->>'adapter_id';
    v_status TEXT := p_provision_row->>'provision_status';
    v_seq INTEGER;
BEGIN
    IF p_submission_id !~ '^source_add_submission:[0-9a-f]{16}$'
       OR v_adapter_id = ''
       OR jsonb_typeof(p_catalog_row) <> 'object'
       OR jsonb_typeof(p_provision_row) <> 'object'
       OR jsonb_typeof(p_smoke_attempt) <> 'object'
       OR v_status NOT IN ('approved_pending_provision', 'provisioned_autoresearch_eligible', 'disabled')
       OR p_catalog_row->>'catalog_id' !~ '^source_catalog:[0-9a-f]{16}$'
       OR p_provision_row->>'provision_ref' !~ '^source_add_provision:[0-9a-f]{16}$'
       OR jsonb_typeof(p_provision_row->'provision_doc') <> 'object'
       OR jsonb_typeof(p_provision_row->'provision_doc'->'provider_registry_entry') <> 'object'
       OR jsonb_typeof(p_provision_row->'provision_doc'->'probe_endpoints') <> 'array'
       OR p_catalog_row::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])'
       OR p_provision_row::TEXT ~* '(sk-or-|sb_secret|service_role|raw_secret|raw_credential|"password"\s*:|"api_key"\s*:\s*"[^"[:space:]])' THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning input is invalid';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-submission:' || p_submission_id, 0)
    );
    SELECT * INTO v_current
    FROM public.research_lab_source_add_submission_current
    WHERE submission_id = p_submission_id;
    IF NOT FOUND THEN RETURN jsonb_build_object('status', 'missing'); END IF;
    IF v_current.adapter_id <> v_adapter_id
       OR p_catalog_row->>'miner_ref' <> v_current.miner_hotkey
       OR p_catalog_row->>'source_name' <> v_current.submission_doc->'manifest'->>'source_name'
       OR p_catalog_row->>'source_kind' <> v_current.submission_doc->'manifest'->>'source_kind'
       OR p_catalog_row->'declared_base_domains' <> v_current.submission_doc->'manifest'->'declared_base_domains'
       OR p_catalog_row->>'source_identity_hash' <> v_current.source_identity_hash
       OR p_catalog_row->>'registry_provider_id' <> p_provision_row->>'registry_provider_id'
       OR p_provision_row->>'adapter_id' <> v_adapter_id
       OR p_provision_row->>'miner_hotkey' <> v_current.miner_hotkey
       OR p_provision_row->>'source_identity_hash' <> v_current.source_identity_hash
       OR p_provision_row->>'submission_id' <> p_submission_id THEN
        RAISE EXCEPTION 'SOURCE_ADD provisioning ownership differs';
    END IF;

    IF v_status = 'disabled' THEN
        IF NOT EXISTS (
            SELECT 1 FROM public.research_lab_source_catalog WHERE adapter_id = v_adapter_id
        ) THEN RETURN jsonb_build_object('status', 'catalog_missing'); END IF;
    ELSE
        SELECT * INTO v_probe
        FROM public.research_lab_source_add_functional_probe_current
        WHERE submission_id = p_submission_id;
        IF NOT FOUND OR v_probe.result_status <> 'passed' THEN
            RETURN jsonb_build_object('status', 'functional_probe_required');
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM public.research_lab_attested_execution_receipts_v2 receipt
            JOIN public.research_lab_attested_business_artifact_links_v2 link
              ON link.receipt_hash = receipt.receipt_hash
            WHERE receipt.receipt_hash = v_probe.receipt_hash
              AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
              AND receipt.receipt_status = 'succeeded'
              AND receipt.output_root = v_probe.business_artifact_hash
              AND link.artifact_kind = 'source_add_functional_probe'
              AND link.artifact_ref = v_probe.attempt_ref
              AND link.artifact_hash = v_probe.business_artifact_hash
        ) THEN RAISE EXCEPTION 'SOURCE_ADD functional receipt is unavailable'; END IF;

        SELECT * INTO v_config
        FROM public.research_lab_source_add_probe_config_current
        WHERE submission_id = p_submission_id AND config_status = 'active';
        IF NOT FOUND OR v_config.config_ref <> v_probe.config_ref THEN
            RETURN jsonb_build_object('status', 'current_probe_config_required');
        END IF;
        IF p_provision_row->'provision_doc'->'provider_registry_entry'->>'base_url' <> v_config.probe_doc->>'base_url'
           OR p_provision_row->'provision_doc'->'provider_registry_entry'->>'auth_kind' <> v_config.probe_doc->>'auth_kind'
           OR COALESCE(p_provision_row->'provision_doc'->'provider_registry_entry'->>'auth_name', '')
              <> COALESCE(v_config.probe_doc->>'auth_name', '')
           OR COALESCE(p_provision_row->'credential_envelope', '{}'::JSONB)
              <> v_config.credential_envelope
           OR COALESCE(p_provision_row->'provision_doc'->'request_headers', '{}'::JSONB)
              <> COALESCE(v_config.probe_doc->'request_headers', '{}'::JSONB)
           OR jsonb_array_length(p_provision_row->'provision_doc'->'probe_endpoints')
              <> jsonb_array_length(v_config.probe_doc->'probes')
           OR EXISTS (
              SELECT 1
              FROM jsonb_array_elements(v_config.probe_doc->'probes') test_probe
              WHERE NOT EXISTS (
                  SELECT 1
                  FROM jsonb_array_elements(p_provision_row->'provision_doc'->'probe_endpoints') endpoint
                  WHERE endpoint->>'method' = test_probe->>'method'
                    AND endpoint->>'path' = test_probe->>'path'
              )
           )
           OR EXISTS (
              SELECT 1
              FROM jsonb_array_elements(
                  p_provision_row->'provision_doc'->'probe_endpoints'
              ) endpoint
              WHERE NOT EXISTS (
                  SELECT 1
                  FROM jsonb_array_elements(v_config.probe_doc->'probes') test_probe
                  WHERE endpoint->>'method' = test_probe->>'method'
                    AND endpoint->>'path' = test_probe->>'path'
              )
           ) THEN
            RETURN jsonb_build_object('status', 'provision_config_differs_from_test');
        END IF;
    END IF;

    IF v_status = 'provisioned_autoresearch_eligible' THEN
        IF p_smoke_attempt = '{}'::JSONB
           OR p_smoke_attempt->>'attempt_ref' !~ '^source_add_probe_attempt:[0-9a-f]{16}$'
           OR p_smoke_attempt->>'work_id' !~ '^source_add_work:[0-9a-f]{16}$'
           OR COALESCE((p_smoke_attempt->>'attempt_number')::INTEGER, 0) NOT BETWEEN 1 AND 20
           OR p_smoke_attempt->>'evaluation_mode' <> 'provisioning_smoke'
           OR p_smoke_attempt->>'config_ref' <> v_config.config_ref
           OR p_smoke_attempt->>'result_status' <> 'passed'
           OR p_smoke_attempt->'result_doc'->>'submission_id' <> p_submission_id
           OR p_smoke_attempt->'result_doc'->>'adapter_id' <> v_adapter_id
           OR p_smoke_attempt->'result_doc'->>'evaluation_mode' <> 'provisioning_smoke'
           OR p_smoke_attempt->'result_doc'->>'config_ref' <> v_config.config_ref
           OR p_smoke_attempt->'result_doc'->>'result_status' <> 'passed'
           OR p_smoke_attempt->'result_doc'->>'route_hash' <> p_smoke_attempt->>'route_hash'
           OR p_smoke_attempt->>'receipt_hash' !~ '^sha256:[0-9a-f]{64}$'
           OR p_smoke_attempt->>'business_artifact_hash' !~ '^sha256:[0-9a-f]{64}$' THEN
            RETURN jsonb_build_object('status', 'smoke_test_required');
        END IF;
        IF NOT EXISTS (
            SELECT 1
            FROM public.research_lab_attested_execution_receipts_v2 receipt
            JOIN public.research_lab_attested_business_artifact_links_v2 link
              ON link.receipt_hash = receipt.receipt_hash
            WHERE receipt.receipt_hash = p_smoke_attempt->>'receipt_hash'
              AND receipt.purpose = 'research_lab.source_add_functional_probe.v2'
              AND receipt.receipt_status = 'succeeded'
              AND receipt.output_root = p_smoke_attempt->>'business_artifact_hash'
              AND link.artifact_kind = 'source_add_provisioning_smoke'
              AND link.artifact_ref = p_smoke_attempt->>'attempt_ref'
              AND link.artifact_hash = p_smoke_attempt->>'business_artifact_hash'
        ) THEN RAISE EXCEPTION 'SOURCE_ADD smoke receipt is unavailable'; END IF;

        INSERT INTO public.research_lab_source_add_functional_probe_attempts (
            attempt_ref, submission_id, adapter_id, work_id, attempt_number,
            evaluation_mode, config_ref, result_status, route_hash, response_hash,
            status_class, content_type, byte_count, duration_ms, retry_after_seconds,
            reason_codes, receipt_hash, business_artifact_hash, result_doc
        ) VALUES (
            p_smoke_attempt->>'attempt_ref', p_submission_id, v_adapter_id,
            p_smoke_attempt->>'work_id',
            (p_smoke_attempt->>'attempt_number')::INTEGER,
            'provisioning_smoke',
            p_smoke_attempt->>'config_ref', 'passed', p_smoke_attempt->>'route_hash',
            COALESCE(p_smoke_attempt->>'response_hash', ''),
            COALESCE(p_smoke_attempt->>'status_class', ''),
            COALESCE(p_smoke_attempt->>'content_type', ''),
            COALESCE((p_smoke_attempt->>'byte_count')::INTEGER, 0),
            COALESCE((p_smoke_attempt->>'duration_ms')::INTEGER, 0),
            COALESCE((p_smoke_attempt->>'retry_after_seconds')::INTEGER, 0),
            COALESCE(p_smoke_attempt->'reason_codes', '[]'::JSONB),
            p_smoke_attempt->>'receipt_hash', p_smoke_attempt->>'business_artifact_hash',
            p_smoke_attempt->'result_doc'
        ) ON CONFLICT (attempt_ref) DO NOTHING;
        SELECT * INTO v_smoke
        FROM public.research_lab_source_add_functional_probe_attempts
        WHERE attempt_ref = p_smoke_attempt->>'attempt_ref';
        IF NOT FOUND OR v_smoke.evaluation_mode <> 'provisioning_smoke'
           OR v_smoke.result_doc <> p_smoke_attempt->'result_doc'
           OR v_smoke.receipt_hash <> p_smoke_attempt->>'receipt_hash' THEN
            RAISE EXCEPTION 'SOURCE_ADD smoke idempotency differs';
        END IF;
    ELSIF p_smoke_attempt <> '{}'::JSONB THEN
        RAISE EXCEPTION 'SOURCE_ADD smoke supplied for non-eligible status';
    END IF;

    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-provision:' || v_adapter_id, 0)
    );
    IF EXISTS (
        SELECT 1 FROM public.research_lab_source_add_provisioning_current
        WHERE registry_provider_id = p_provision_row->>'registry_provider_id'
          AND adapter_id <> v_adapter_id
    ) THEN RETURN jsonb_build_object('status', 'registry_provider_conflict'); END IF;

    SELECT * INTO v_catalog FROM public.research_lab_source_catalog
    WHERE adapter_id = v_adapter_id;
    IF NOT FOUND THEN
        IF v_status = 'disabled' THEN
            RETURN jsonb_build_object('status', 'catalog_missing');
        END IF;
        INSERT INTO public.research_lab_source_catalog (
            catalog_id, adapter_id, miner_ref, source_name, source_kind,
            declared_base_domains, registry_provider_id, measured_trial_yield,
            accepted_at, catalog_doc, source_identity_hash
        ) VALUES (
            p_catalog_row->>'catalog_id', v_adapter_id, p_catalog_row->>'miner_ref',
            p_catalog_row->>'source_name', p_catalog_row->>'source_kind',
            p_catalog_row->'declared_base_domains', p_catalog_row->>'registry_provider_id',
            0.0, NOW(), p_catalog_row->'catalog_doc', p_catalog_row->>'source_identity_hash'
        );
        v_catalog_id := p_catalog_row->>'catalog_id';
    ELSE
        v_catalog_id := v_catalog.catalog_id;
        IF v_catalog_id <> p_catalog_row->>'catalog_id'
           OR v_catalog.miner_ref <> p_catalog_row->>'miner_ref'
           OR v_catalog.source_name <> p_catalog_row->>'source_name'
           OR v_catalog.source_kind <> p_catalog_row->>'source_kind'
           OR to_jsonb(v_catalog.declared_base_domains) <> p_catalog_row->'declared_base_domains'
           OR v_catalog.registry_provider_id <> p_catalog_row->>'registry_provider_id'
           OR v_catalog.source_identity_hash <> p_catalog_row->>'source_identity_hash' THEN
            RAISE EXCEPTION 'SOURCE_ADD catalog identity differs';
        END IF;
    END IF;
    SELECT COALESCE(MAX(seq), -1) + 1 INTO v_seq
    FROM public.research_lab_source_add_provisioning_events WHERE adapter_id = v_adapter_id;
    IF EXISTS (
        SELECT 1 FROM public.research_lab_source_add_provisioning_events
        WHERE provision_ref = p_provision_row->>'provision_ref'
    ) THEN
        RETURN jsonb_build_object(
            'status', 'already_provisioned', 'catalog_id', v_catalog_id,
            'provision_ref', p_provision_row->>'provision_ref'
        );
    END IF;
    INSERT INTO public.research_lab_source_add_provisioning_events (
        provision_ref, catalog_id, submission_id, adapter_id, miner_hotkey,
        source_identity_hash, registry_provider_id, provision_status, seq,
        provision_doc, credential_envelope
    ) VALUES (
        p_provision_row->>'provision_ref', v_catalog_id, p_submission_id, v_adapter_id,
        p_provision_row->>'miner_hotkey', p_provision_row->>'source_identity_hash',
        p_provision_row->>'registry_provider_id', p_provision_row->>'provision_status',
        v_seq, p_provision_row->'provision_doc',
        COALESCE(p_provision_row->'credential_envelope', '{}'::JSONB)
    );
    IF v_status <> 'disabled' AND v_current.stage <> 'accepted' THEN
        INSERT INTO public.research_lab_source_add_submissions (
            submission_id, adapter_id, miner_hotkey, stage, seq, submission_doc,
            precheck_status, precheck_doc, source_identity_hash, source_identity_version
        ) VALUES (
            p_submission_id, v_adapter_id, v_current.miner_hotkey, 'accepted',
            (SELECT COALESCE(MAX(seq), -1) + 1
             FROM public.research_lab_source_add_submissions
             WHERE submission_id = p_submission_id),
            v_current.submission_doc || jsonb_build_object('stage', 'accepted'),
            v_current.precheck_status, v_current.precheck_doc,
            v_current.source_identity_hash, v_current.source_identity_version
        );
    END IF;
    RETURN jsonb_build_object(
        'status', 'provisioned', 'catalog_id', v_catalog_id, 'seq', v_seq,
        'provision_ref', p_provision_row->>'provision_ref'
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_finalize_provision_smoke(
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
    v_result JSONB;
BEGIN
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

-- The original V2 migration omitted live coordinator purposes already used by
-- SOURCE_ADD. Replace only the role/purpose CHECK with a strict superset.
DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'public.research_lab_attested_execution_receipts_v2'::REGCLASS
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE '%gateway_coordinator%'
          AND pg_get_constraintdef(oid) LIKE '%purpose%'
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_attested_execution_receipts_v2 DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_attested_execution_receipts_v2
    ADD CONSTRAINT research_lab_attested_execution_receipts_v2_role_purpose_check
    CHECK (
        (role = 'gateway_coordinator' AND purpose IN (
            'research_lab.admission.v2',
            'research_lab.provider_evidence.v2',
            'research_lab.provider_outcome_state.v2',
            'leadpoet.artifact_persistence.v2',
            'research_lab.ranking.v2',
            'research_lab.promotion_decision.v2',
            'research_lab.reward_decision.v2',
            'research_lab.allocation.v2',
            'research_lab.champion_input.v2',
            'research_lab.reimbursement_input.v2',
            'research_lab.source_add_reward_input.v2',
            'research_lab.source_add_provenance.v2',
            'research_lab.source_add_functional_probe.v2',
            'research_lab.source_add_catalog_snapshot.v2',
            'research_lab.source_add_credential.v2',
            'research_lab.provider_outcome_snapshot.v2',
            'research_lab.openrouter_credential.v2',
            'research_lab.openrouter_credit_preflight.v2',
            'research_lab.active_private_model.v2',
            'research_lab.sourcing_input.v2',
            'research_lab.fulfillment_input.v2',
            'research_lab.leaderboard_input.v2',
            'research_lab.ban_input.v2',
            'research_lab.anomaly_adjustment_input.v2',
            'gateway.weights.publication.v2'
        )) OR
        (role = 'gateway_scoring' AND purpose IN (
            'research_lab.private_model_run.v2',
            'research_lab.candidate_model_run.v2',
            'research_lab.provider_evidence_tape.v2',
            'research_lab.candidate_test.v2',
            'research_lab.company_score.v2',
            'research_lab.provider_preflight.v2',
            'research_lab.candidate_score.v2',
            'research_lab.baseline_score.v2',
            'research_lab.benchmark.v2',
            'research_lab.rebenchmark.v2',
            'research_lab.confirmation_score.v2',
            'research_lab.source_add_judge.v2',
            'qualification.lead_decision.v2',
            'qualification.email_evidence.v2',
            'qualification.sourcing_epoch.v2'
        )) OR
        (role = 'gateway_autoresearch' AND purpose IN (
            'research_lab.source_inspection.v2',
            'research_lab.research_plan.v2',
            'research_lab.patch_draft.v2',
            'research_lab.patch_validation.v2',
            'research_lab.candidate_test.v2',
            'research_lab.candidate_build.v2',
            'research_lab.candidate_decision.v2',
            'research_lab.stale_parent_repair.v2',
            'research_lab.checkpoint.v2',
            'research_lab.openrouter_guard.v2'
        )) OR
        (role = 'validator_weights' AND purpose IN (
            'validator.weight_snapshot.v2',
            'validator.weights.computed.v2',
            'validator.chain_state.v2',
            'validator.metagraph_state.v2',
            'validator.burn_ownership.v2',
            'validator.feature_flags.v2',
            'validator.constants.v2',
            'validator.hotkey_signature.v2',
            'validator.serve_axon_extrinsic.v2',
            'validator.set_weights_extrinsic.v2',
            'validator.weights.finalized.v2'
        ))
    ) NOT VALID;
ALTER TABLE public.research_lab_attested_execution_receipts_v2
    VALIDATE CONSTRAINT research_lab_attested_execution_receipts_v2_role_purpose_check;

-- Service-role only access. Queue/control/slot tables are operational and are
-- mutated only through the transaction functions above.
REVOKE ALL ON TABLE public.research_lab_source_add_identity_events FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_identity_current FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_probe_config_events FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_probe_config_current FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_functional_probe_attempts FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_functional_probe_current FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_provisioning_smoke_current FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_work_items FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_reward_intents FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_reward_slots FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_source_add_control FROM PUBLIC, anon, authenticated;

GRANT SELECT, INSERT ON TABLE public.research_lab_source_add_identity_events TO service_role;
GRANT SELECT ON TABLE public.research_lab_source_add_identity_current TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_source_add_probe_config_events TO service_role;
GRANT SELECT ON TABLE public.research_lab_source_add_probe_config_current TO service_role;
GRANT SELECT, INSERT ON TABLE public.research_lab_source_add_functional_probe_attempts TO service_role;
GRANT SELECT ON TABLE public.research_lab_source_add_functional_probe_current TO service_role;
GRANT SELECT ON TABLE public.research_lab_source_add_provisioning_smoke_current TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_source_add_work_items TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_source_add_reward_intents TO service_role;
GRANT SELECT, INSERT, UPDATE ON TABLE public.research_lab_source_add_reward_slots TO service_role;
GRANT SELECT, UPDATE ON TABLE public.research_lab_source_add_control TO service_role;

ALTER TABLE public.research_lab_source_add_identity_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_probe_config_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_functional_probe_attempts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_work_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_reward_intents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_reward_slots ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_control ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE
    table_name TEXT;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'research_lab_source_add_identity_events',
        'research_lab_source_add_probe_config_events',
        'research_lab_source_add_functional_probe_attempts',
        'research_lab_source_add_work_items',
        'research_lab_source_add_reward_intents',
        'research_lab_source_add_reward_slots',
        'research_lab_source_add_control'
    ]
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON public.%I', table_name || '_service_all', table_name);
        EXECUTE format(
            'CREATE POLICY %I ON public.%I FOR ALL TO service_role USING (true) WITH CHECK (true)',
            table_name || '_service_all', table_name
        );
    END LOOP;
END;
$$;

REVOKE ALL ON FUNCTION public.research_lab_source_add_admit(JSONB, TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_begin_provider_execution(TEXT, UUID) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_claim_work(TEXT, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finish_work(TEXT, UUID, TEXT, TEXT, JSONB, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB, JSONB, TIMESTAMPTZ, BOOLEAN) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_configure_probe(TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_requeue_provenance(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_set_paused(BOOLEAN, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_reserve_leg1_slot(TEXT, TEXT, UUID, INTEGER, INTEGER) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_leg1(TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke(TEXT, TEXT, TEXT, TEXT, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision(TEXT, JSONB, JSONB, JSONB) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_finalize_provision_smoke(TEXT, UUID, TEXT, JSONB, JSONB, JSONB) FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.research_lab_source_add_admit(JSONB, TEXT, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_begin_provider_execution(TEXT, UUID) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_claim_work(TEXT, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finish_work(TEXT, UUID, TEXT, TEXT, JSONB, TEXT, JSONB, JSONB, JSONB, JSONB, JSONB, JSONB, TIMESTAMPTZ, BOOLEAN) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_configure_probe(TEXT, TEXT, JSONB, JSONB, TEXT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_requeue_provenance(TEXT, TEXT, TEXT, TEXT, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_set_paused(BOOLEAN, TEXT, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_reserve_leg1_slot(TEXT, TEXT, UUID, INTEGER, INTEGER) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_leg1(TEXT, TEXT, UUID, UUID, INTEGER, JSONB, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_enqueue_provision_smoke(TEXT, TEXT, TEXT, TEXT, JSONB, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_provision(TEXT, JSONB, JSONB, JSONB) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_finalize_provision_smoke(TEXT, UUID, TEXT, JSONB, JSONB, JSONB) TO service_role;

COMMENT ON TABLE public.research_lab_source_add_work_items IS
    'SOURCE_ADD-only leased work queue; does not share hosted or scoring worker capacity.';
COMMENT ON TABLE public.research_lab_source_add_functional_probe_attempts IS
    'Append-only bounded V2 functional API probe summaries; response bodies and credentials are forbidden.';
COMMENT ON TABLE public.research_lab_source_add_reward_slots IS
    'UTC-day SOURCE_ADD Leg 1 FIFO reservation slots; legacy and functional rewards share the daily cap.';

COMMIT;
