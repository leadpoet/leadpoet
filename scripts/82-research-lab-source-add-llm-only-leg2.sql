-- SOURCE_ADD Leg 2 is authorized only by the final LLM attribution judge.
--
-- This migration is intentionally fail-closed. If a historical Leg 2 row has
-- only shadow/ablation evidence, the transaction aborts before constraints are
-- changed; no reward row is rewritten or deleted.

BEGIN;

SET LOCAL lock_timeout = '5s';

DO $$
DECLARE
    legacy_leg2_count BIGINT;
    constraint_row RECORD;
BEGIN
    SELECT COUNT(*)
      INTO legacy_leg2_count
      FROM public.research_lab_source_add_reward_obligations
     WHERE leg = 2
       AND NOT (
           trigger_evidence_doc @> '{"llm_judge_passed": true}'::JSONB
       );

    IF legacy_leg2_count > 0 THEN
        RAISE EXCEPTION
            'SOURCE_ADD Leg 2 migration aborted: % row(s) lack llm_judge_passed=true',
            legacy_leg2_count;
    END IF;

    FOR constraint_row IN
        SELECT c.conname
          FROM pg_constraint c
         WHERE c.conrelid =
               'public.research_lab_source_add_reward_obligations'::REGCLASS
           AND c.contype = 'c'
           AND (
               pg_get_constraintdef(c.oid) ILIKE '%shadow_window_passed%'
               OR pg_get_constraintdef(c.oid) ILIKE '%ablation_passed%'
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
    ADD CONSTRAINT research_lab_source_add_reward_leg2_llm_only_check
    CHECK (
        leg = 1
        OR trigger_evidence_doc @> '{"llm_judge_passed": true}'::JSONB
    ) NOT VALID;

ALTER TABLE public.research_lab_source_add_reward_obligations
    VALIDATE CONSTRAINT research_lab_source_add_reward_leg2_llm_only_check;

COMMENT ON CONSTRAINT research_lab_source_add_reward_leg2_llm_only_check
    ON public.research_lab_source_add_reward_obligations IS
    'Leg 2 requires llm_judge_passed=true; shadow-window or ablation evidence never authorizes a reward.';

COMMIT;
