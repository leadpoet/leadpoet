-- Align the SOURCE_ADD Leg 2 database gate with the LLM-judge payout path.
-- Keep the legacy shadow/ablation shape valid for already-created rows.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    constraint_row RECORD;
BEGIN
    FOR constraint_row IN
        SELECT c.conname
        FROM pg_constraint c
        WHERE c.conrelid = 'public.research_lab_source_add_reward_obligations'::REGCLASS
          AND c.contype = 'c'
          AND (
              pg_get_constraintdef(c.oid) ILIKE '%shadow_window_passed%'
              OR pg_get_constraintdef(c.oid) ILIKE '%llm_judge_passed%'
          )
    LOOP
        EXECUTE format(
            'ALTER TABLE public.research_lab_source_add_reward_obligations DROP CONSTRAINT %I',
            constraint_row.conname
        );
    END LOOP;
END;
$$;

ALTER TABLE public.research_lab_source_add_reward_obligations
    ADD CONSTRAINT research_lab_source_add_reward_leg2_evidence_check
    CHECK (
        leg = 1
        OR trigger_evidence_doc @> '{"llm_judge_passed": true}'::JSONB
        OR (
            trigger_evidence_doc @> '{"shadow_window_passed": true}'::JSONB
            AND trigger_evidence_doc @> '{"ablation_passed": true}'::JSONB
        )
    ) NOT VALID;

ALTER TABLE public.research_lab_source_add_reward_obligations
    VALIDATE CONSTRAINT research_lab_source_add_reward_leg2_evidence_check;

COMMENT ON CONSTRAINT research_lab_source_add_reward_leg2_evidence_check
    ON public.research_lab_source_add_reward_obligations IS
    'Leg 2 requires the current LLM final judge or legacy shadow-window and ablation evidence.';

COMMIT;
