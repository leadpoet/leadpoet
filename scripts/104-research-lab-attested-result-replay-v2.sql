-- Durable replay for public, enclave-signed V2 allocation and weight inputs.
-- Apply after migration 103.
--
-- The execution receipt remains authoritative. This append-only table retains
-- only the sanitized result and the exact artifact commitments needed to prove
-- that a same-release enclave restart is replaying that receipt without a
-- second provider operation.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS public.research_lab_attested_execution_results_v2 (
    receipt_hash        TEXT        PRIMARY KEY
                                    REFERENCES public.research_lab_attested_execution_receipts_v2(receipt_hash)
                                    ON DELETE RESTRICT,
    schema_version      TEXT        NOT NULL CHECK (
                                    schema_version = 'leadpoet.attested_execution_result.v2'
                                ),
    role                TEXT        NOT NULL CHECK (role = 'gateway_coordinator'),
    operation           TEXT        NOT NULL CHECK (
                                    operation IN (
                                        'research_lab_allocation',
                                        'attest_weight_input'
                                    )
                                ),
    purpose             TEXT        NOT NULL CHECK (
                                    purpose IN (
                                        'research_lab.allocation.v2',
                                        'research_lab.champion_input.v2',
                                        'research_lab.reimbursement_input.v2',
                                        'research_lab.source_add_reward_input.v2',
                                        'research_lab.sourcing_input.v2',
                                        'research_lab.fulfillment_input.v2',
                                        'research_lab.leaderboard_input.v2',
                                        'research_lab.ban_input.v2',
                                        'research_lab.anomaly_adjustment_input.v2'
                                    )
                                ),
    job_id              TEXT        NOT NULL,
    epoch_id            INTEGER     NOT NULL CHECK (epoch_id >= 0),
    sequence            INTEGER     NOT NULL CHECK (sequence >= 0),
    release_hash        TEXT        NOT NULL CHECK (release_hash ~ '^sha256:[0-9a-f]{64}$'),
    input_root          TEXT        NOT NULL CHECK (input_root ~ '^sha256:[0-9a-f]{64}$'),
    output_root         TEXT        NOT NULL CHECK (output_root ~ '^sha256:[0-9a-f]{64}$'),
    artifact_root       TEXT        NOT NULL CHECK (artifact_root ~ '^sha256:[0-9a-f]{64}$'),
    result_hash         TEXT        NOT NULL CHECK (result_hash ~ '^sha256:[0-9a-f]{64}$'),
    artifact_hashes     JSONB       NOT NULL CHECK (jsonb_typeof(artifact_hashes) = 'array'),
    result_doc          JSONB       NOT NULL CHECK (
                                    jsonb_typeof(result_doc) = 'object'
                                    AND result_doc::TEXT !~* '(sk-or-|sb_secret|service_role|openrouter_api_key|scrapingdog_api_key|exa_api_key|deepline_api_key|raw_secret|private_repo|judge_prompt|hidden_icp|provider_output|request_body|response_body|authorization|proxy-authorization|://[^/]+:[^/@]+@)'
                                ),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (role, purpose, job_id),
    CHECK (
        (operation = 'research_lab_allocation'
         AND purpose = 'research_lab.allocation.v2')
        OR operation = 'attest_weight_input'
    )
);

ALTER TABLE public.research_lab_attested_execution_results_v2
    ENABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_research_lab_attested_results_v2_epoch
    ON public.research_lab_attested_execution_results_v2(
        epoch_id DESC, purpose, created_at DESC
    );

DO $$
DECLARE
    trigger_name TEXT := 'prevent_v2_' || substring(
        md5('research_lab_attested_execution_results_v2') FROM 1 FOR 16
    ) || '_mutation';
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgrelid =
              'public.research_lab_attested_execution_results_v2'::regclass
          AND tgname = trigger_name
          AND NOT tgisinternal
    ) THEN
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON '
            'public.research_lab_attested_execution_results_v2 '
            'FOR EACH ROW EXECUTE FUNCTION '
            'public.prevent_research_lab_attested_v2_mutation()',
            trigger_name
        );
    END IF;
END;
$$;

REVOKE ALL ON TABLE public.research_lab_attested_execution_results_v2
    FROM PUBLIC, anon, authenticated, service_role;
GRANT SELECT, INSERT
    ON TABLE public.research_lab_attested_execution_results_v2
    TO service_role;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_attested_execution_results_v2;
CREATE POLICY service_role_read
    ON public.research_lab_attested_execution_results_v2
    FOR SELECT TO service_role USING (true);
DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_attested_execution_results_v2;
CREATE POLICY service_role_insert
    ON public.research_lab_attested_execution_results_v2
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_attested_execution_results_v2 IS
    'Append-only sanitized V2 results bound to durable enclave receipts for exact same-release restart replay.';

NOTIFY pgrst, 'reload schema';

COMMIT;
