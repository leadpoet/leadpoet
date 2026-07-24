-- Fulfillment finalization correctness fence.
--
-- 1. Score writes lock the request row and reject terminal/finalizing states.
-- 2. Finalization locks the same row and succeeds only when the score count
--    and latest score timestamp still match the set used for consensus.
-- 3. Newly-added score detail columns are persisted inside the score RPC
--    transaction instead of relying only on the gateway's compatibility
--    patch after the RPC commits.

BEGIN;

ALTER TABLE public.fulfillment_requests
    DROP CONSTRAINT IF EXISTS fulfillment_requests_status_check;

ALTER TABLE public.fulfillment_requests
    ADD CONSTRAINT fulfillment_requests_status_check
    CHECK (
        status IN (
            'pending',
            'open',
            'continued_open',
            'commit_closed',
            'scoring',
            'finalizing',
            'fulfilled',
            'partially_fulfilled',
            'recycled',
            'expired'
        )
    );

CREATE OR REPLACE FUNCTION public.fulfillment_upsert_scores(
    p_scores jsonb,
    p_validator_hotkey text
)
RETURNS void
LANGUAGE plpgsql
AS $function$
DECLARE
    v_score jsonb;
    v_request_id uuid;
    v_score_request_id uuid;
    v_request_status text;
BEGIN
    IF jsonb_typeof(p_scores) IS DISTINCT FROM 'array'
       OR jsonb_array_length(p_scores) = 0 THEN
        RETURN;
    END IF;

    v_request_id := (p_scores->0->>'request_id')::uuid;

    SELECT status
      INTO v_request_status
      FROM public.fulfillment_requests
     WHERE request_id = v_request_id
     FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'FULFILLMENT_REQUEST_NOT_FOUND request=%',
            v_request_id;
    END IF;

    IF v_request_status NOT IN ('scoring', 'partially_fulfilled') THEN
        RAISE EXCEPTION
            'FULFILLMENT_SCORE_WINDOW_CLOSED request=% status=%',
            v_request_id,
            v_request_status;
    END IF;

    FOR v_score IN SELECT * FROM jsonb_array_elements(p_scores) LOOP
        v_score_request_id := (v_score->>'request_id')::uuid;
        IF v_score_request_id IS DISTINCT FROM v_request_id THEN
            RAISE EXCEPTION
                'FULFILLMENT_SCORE_REQUEST_MISMATCH expected=% actual=%',
                v_request_id,
                v_score_request_id;
        END IF;

        INSERT INTO public.fulfillment_scores (
            score_id,
            request_id,
            submission_id,
            miner_hotkey,
            validator_hotkey,
            lead_id,
            icp_fit,
            decision_maker,
            intent_signal_raw,
            intent_signal_final,
            intent_decay_multiplier,
            final_score,
            tier1_passed,
            tier2_passed,
            email_verified,
            person_verified,
            company_verified,
            rep_score,
            failure_reason,
            failure_detail,
            intent_signals_detail,
            attribute_verification,
            scored_at,
            all_fabricated
        ) VALUES (
            gen_random_uuid(),
            v_score_request_id,
            (v_score->>'submission_id')::uuid,
            v_score->>'miner_hotkey',
            p_validator_hotkey,
            v_score->>'lead_id',
            NULL,
            NULL,
            (v_score->>'intent_signal_raw')::double precision,
            (v_score->>'intent_signal_final')::double precision,
            (v_score->>'intent_decay_multiplier')::double precision,
            (v_score->>'final_score')::double precision,
            COALESCE((v_score->>'tier1_passed')::boolean, false),
            COALESCE((v_score->>'tier2_passed')::boolean, false),
            COALESCE((v_score->>'email_verified')::boolean, false),
            COALESCE((v_score->>'person_verified')::boolean, false),
            COALESCE((v_score->>'company_verified')::boolean, false),
            COALESCE((v_score->>'rep_score')::double precision, 0.0),
            v_score->>'failure_reason',
            v_score->>'failure_detail',
            NULLIF(v_score->'intent_signals_detail', 'null'::jsonb),
            NULLIF(v_score->'attribute_verification', 'null'::jsonb),
            clock_timestamp(),
            COALESCE((v_score->>'all_fabricated')::boolean, false)
        )
        ON CONFLICT (
            request_id,
            submission_id,
            lead_id,
            validator_hotkey
        ) DO UPDATE SET
            intent_signal_raw = EXCLUDED.intent_signal_raw,
            intent_signal_final = EXCLUDED.intent_signal_final,
            intent_decay_multiplier = EXCLUDED.intent_decay_multiplier,
            final_score = EXCLUDED.final_score,
            tier1_passed = EXCLUDED.tier1_passed,
            tier2_passed = EXCLUDED.tier2_passed,
            email_verified = EXCLUDED.email_verified,
            person_verified = EXCLUDED.person_verified,
            company_verified = EXCLUDED.company_verified,
            rep_score = EXCLUDED.rep_score,
            failure_reason = EXCLUDED.failure_reason,
            failure_detail = COALESCE(
                EXCLUDED.failure_detail,
                fulfillment_scores.failure_detail
            ),
            intent_signals_detail = COALESCE(
                EXCLUDED.intent_signals_detail,
                fulfillment_scores.intent_signals_detail
            ),
            attribute_verification = COALESCE(
                EXCLUDED.attribute_verification,
                fulfillment_scores.attribute_verification
            ),
            scored_at = EXCLUDED.scored_at,
            all_fabricated = EXCLUDED.all_fabricated;
    END LOOP;
END;
$function$;

CREATE OR REPLACE FUNCTION public.fulfillment_claim_finalization(
    p_request_id uuid,
    p_expected_score_count bigint,
    p_expected_latest_scored_at timestamptz
)
RETURNS boolean
LANGUAGE plpgsql
AS $function$
DECLARE
    v_request_status text;
    v_score_count bigint;
    v_latest_scored_at timestamptz;
BEGIN
    SELECT status
      INTO v_request_status
      FROM public.fulfillment_requests
     WHERE request_id = p_request_id
     FOR UPDATE;

    IF NOT FOUND THEN
        RETURN false;
    END IF;

    IF v_request_status = 'finalizing' THEN
        RETURN true;
    END IF;

    IF v_request_status NOT IN ('scoring', 'partially_fulfilled') THEN
        RETURN false;
    END IF;

    SELECT count(*), max(scored_at)
      INTO v_score_count, v_latest_scored_at
      FROM public.fulfillment_scores
     WHERE request_id = p_request_id;

    IF v_score_count IS DISTINCT FROM p_expected_score_count
       OR v_latest_scored_at IS DISTINCT FROM p_expected_latest_scored_at THEN
        RETURN false;
    END IF;

    UPDATE public.fulfillment_requests
       SET status = 'finalizing'
     WHERE request_id = p_request_id;

    RETURN true;
END;
$function$;

REVOKE ALL ON FUNCTION public.fulfillment_claim_finalization(
    uuid,
    bigint,
    timestamptz
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.fulfillment_claim_finalization(
    uuid,
    bigint,
    timestamptz
) TO service_role;

COMMIT;
