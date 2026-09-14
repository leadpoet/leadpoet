-- Retire the five-ICP confirmation stage. Integrity rounds publish and promote
-- from the complete twenty-ICP score under the existing +1 and cost guards.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_247$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_deepline_cost_reconciliation_schema_v1()'
     ) IS NULL
     OR pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
       'public.lab_arena_deepline_cost_reconciliation_schema_v1()'
     )) NOT LIKE '%''version'', 247%' THEN
    RAISE EXCEPTION 'apply migration 247 before migration 248';
  END IF;
END;
$requires_247$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Production has no historical stage-three work. Fail if that changes during
-- rollout rather than guessing how to rewrite accepted work or provider cost.
DO $no_confirmation_work$
BEGIN
  IF EXISTS (
       SELECT 1 FROM public.lab_arena_rounds
       WHERE status IN (
         'stage3', 'stage3_closed', 'stage3_scoring',
         'stage3_judged', 'confirmed'
       )
     )
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE stage = 3)
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger WHERE stage = 3) THEN
    RAISE EXCEPTION 'lab_arena_confirmation_work_exists';
  END IF;
END;
$no_confirmation_work$;

-- Keep published and cancelled documents, hashes, and inert confirmation
-- columns byte-for-byte unchanged. Only active round configuration loses the
-- retired schedule keys; all other cutoffs and policies remain frozen.
ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_rounds_write_once;
ALTER TABLE public.lab_arena_rounds
  DISABLE TRIGGER lab_arena_integrity_round_guard;
UPDATE public.lab_arena_rounds
SET configuration_doc = pg_catalog.jsonb_set(
      configuration_doc,
      '{schedule}',
      (configuration_doc -> 'schedule')
        - 'stage_3_start' - 'stage_3_close' - 'stage_3_scoring_close'
    ),
    updated_at = pg_catalog.clock_timestamp()
WHERE status NOT IN ('published', 'cancelled')
  AND configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1'
  AND configuration_doc -> 'schedule'
      ?| ARRAY['stage_3_start', 'stage_3_close', 'stage_3_scoring_close'];
ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_integrity_round_guard;
ALTER TABLE public.lab_arena_rounds
  ENABLE TRIGGER lab_arena_rounds_write_once;

ALTER TABLE public.lab_arena_rounds
  DROP CONSTRAINT IF EXISTS lab_arena_rounds_status_check;
ALTER TABLE public.lab_arena_rounds
  ADD CONSTRAINT lab_arena_rounds_status_check CHECK (status IN (
    'open', 'committed', 'stage1', 'stage1_closed', 'stage1_scoring',
    'stage1_judged', 'stage1_scored', 'stage2', 'stage2_closed',
    'stage2_scoring', 'stage2_judged', 'scored', 'published', 'cancelled'
  ));
ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_stage_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_stage_check CHECK (stage IN (1, 2));
ALTER TABLE public.lab_arena_ledger
  DROP CONSTRAINT IF EXISTS lab_arena_ledger_stage_check;
ALTER TABLE public.lab_arena_ledger
  ADD CONSTRAINT lab_arena_ledger_stage_check
  CHECK (stage IS NULL OR stage IN (1, 2));

CREATE OR REPLACE FUNCTION public.lab_arena_integrity_round_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $round_guard$
DECLARE
  v_integrity BOOLEAN;
BEGIN
  IF NEW.configuration_doc ? 'integrity_policy'
     AND NEW.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1' THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_invalid'
      USING ERRCODE = '23514';
  END IF;
  v_integrity := COALESCE(
    NEW.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1', FALSE
  );
  IF NEW.configuration_doc ? 'contact_policy' AND (
       NEW.configuration_doc ->> 'contact_policy' IS DISTINCT FROM 'contacts_v1'
       OR NOT v_integrity) THEN
    RAISE EXCEPTION 'lab_arena_contact_policy_invalid' USING ERRCODE = '23514';
  END IF;
  IF NOT v_integrity AND NEW.configuration_doc #>>
       '{scorer_policy,scoring_adapter_version}' = 'qualification_contacts_v3' THEN
    RAISE EXCEPTION 'lab_arena_contact_policy_required' USING ERRCODE = '23514';
  END IF;
  IF v_integrity AND (
       NEW.configuration_doc #>> '{scorer_policy,scoring_adapter_version}'
         IS DISTINCT FROM (
           CASE WHEN NEW.configuration_doc ->> 'contact_policy' = 'contacts_v1'
             THEN 'qualification_contacts_v3'
             ELSE 'qualification_integrity_v2'
           END
         )
       OR NOT (NEW.configuration_doc ? 'cost_per_company_microusd')
     ) THEN
    RAISE EXCEPTION 'lab_arena_integrity_configuration_invalid'
      USING ERRCODE = '23514';
  END IF;
  IF TG_OP = 'INSERT'
     AND NEW.configuration_doc -> 'schedule'
       ?| ARRAY['stage_3_start', 'stage_3_close', 'stage_3_scoring_close'] THEN
    RAISE EXCEPTION 'lab_arena_confirmation_retired' USING ERRCODE = '23514';
  END IF;
  IF TG_OP = 'UPDATE'
     AND NEW.configuration_doc -> 'schedule'
       ?| ARRAY['stage_3_start', 'stage_3_close', 'stage_3_scoring_close']
     AND (
       NEW.configuration_doc #> '{schedule,stage_3_start}'
         IS DISTINCT FROM OLD.configuration_doc #> '{schedule,stage_3_start}'
       OR NEW.configuration_doc #> '{schedule,stage_3_close}'
         IS DISTINCT FROM OLD.configuration_doc #> '{schedule,stage_3_close}'
       OR NEW.configuration_doc #> '{schedule,stage_3_scoring_close}'
         IS DISTINCT FROM OLD.configuration_doc #> '{schedule,stage_3_scoring_close}'
     ) THEN
    RAISE EXCEPTION 'lab_arena_confirmation_retired' USING ERRCODE = '23514';
  END IF;
  IF TG_OP = 'INSERT' AND (
       NEW.confirmation_bank_ref IS NOT NULL
       OR NEW.confirmation_bank_hash IS NOT NULL
       OR NEW.confirmation_cohort IS NOT NULL
       OR NEW.stage3_scoring_plan_doc IS NOT NULL
     ) THEN
    RAISE EXCEPTION 'lab_arena_confirmation_retired' USING ERRCODE = '23514';
  END IF;
  IF TG_OP = 'UPDATE' AND (
       NEW.confirmation_bank_ref IS DISTINCT FROM OLD.confirmation_bank_ref
       OR NEW.confirmation_bank_hash IS DISTINCT FROM OLD.confirmation_bank_hash
       OR NEW.confirmation_cohort IS DISTINCT FROM OLD.confirmation_cohort
       OR NEW.stage3_scoring_plan_doc IS DISTINCT FROM OLD.stage3_scoring_plan_doc
     ) THEN
    RAISE EXCEPTION 'lab_arena_integrity_state_write_once'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$round_guard$;
ALTER FUNCTION public.lab_arena_integrity_round_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_integrity_round_guard_v1() FROM PUBLIC;

CREATE OR REPLACE FUNCTION public.lab_arena_integrity_run_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $run_guard$
DECLARE
  v_round public.lab_arena_rounds;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = NEW.round_id;
  IF NOT FOUND THEN RETURN NEW; END IF;
  IF (NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND 9)
     OR (NEW.stage = 2 AND NEW.icp_position NOT BETWEEN 10 AND
           CASE WHEN v_round.configuration_doc ->> 'integrity_policy'
                       = 'arena_integrity_v1' THEN 19 ELSE 29 END) THEN
    RAISE EXCEPTION 'lab_arena_run_position_invalid' USING ERRCODE = '23514';
  END IF;
  IF v_round.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1' THEN
    IF NEW.per_icp_score IS NOT NULL
       AND NOT public.lab_arena__qualification_doc_valid(
         NEW.qualification_doc
       ) THEN
      RAISE EXCEPTION 'lab_arena_qualification_receipt_required'
        USING ERRCODE = '23514';
    END IF;
  ELSIF NEW.qualification_doc IS NOT NULL THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required'
      USING ERRCODE = '23514';
  END IF;
  IF NEW.qualification_doc IS NOT NULL THEN
    IF EXISTS (
      SELECT 1
      FROM pg_catalog.jsonb_array_elements(
        NEW.qualification_doc -> 'companies'
      ) AS item
      WHERE (item ? 'contact_qualified') IS DISTINCT FROM COALESCE(
        v_round.configuration_doc ->> 'contact_policy' = 'contacts_v1', FALSE
      )
    ) THEN
      RAISE EXCEPTION 'lab_arena_contact_receipt_policy_mismatch'
        USING ERRCODE = '23514';
    END IF;
    IF v_round.configuration_doc ->> 'contact_policy' = 'contacts_v1'
       AND NEW.per_icp_score > 0 AND NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(
           NEW.qualification_doc -> 'companies'
         ) AS item
         WHERE item ->> 'company_qualified' = 'true'
           AND item ->> 'contact_qualified' = 'true'
           AND item ->> 'duplicate_company' = 'false'
       ) THEN
      RAISE EXCEPTION 'lab_arena_contact_credit_invalid'
        USING ERRCODE = '23514';
    END IF;
  END IF;
  IF TG_OP = 'UPDATE' AND OLD.qualification_doc IS NOT NULL
     AND NEW.qualification_doc IS DISTINCT FROM OLD.qualification_doc THEN
    RAISE EXCEPTION 'lab_arena_qualification_receipt_write_once'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$run_guard$;
ALTER FUNCTION public.lab_arena_integrity_run_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_integrity_run_guard_v1() FROM PUBLIC;

-- Restore all shared RPCs to two-stage bounds without replacing later fixes.
DO $restore_shared_stage_functions$
DECLARE
  v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_close_stage(text,smallint)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(
    v_definition, 'p_stage NOT IN (1, 2, 3)', 'p_stage NOT IN (1, 2)'
  );
  v_definition := pg_catalog.replace(v_definition,
    $old$OR (p_stage = 3 AND v_round.status IN ('stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed', 'published')) OR (p_stage = 2 AND v_round.status IN ($old$,
    $new$OR (p_stage = 2 AND v_round.status IN ($new$
  );
  v_definition := pg_catalog.replace(v_definition,
    $old$'scored', 'stage3', 'stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed', 'published'$old$,
    $new$'scored', 'published'$new$
  );
  EXECUTE v_definition;

  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_close_scoring(text,smallint)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(
    v_definition, 'p_stage NOT IN (1, 2, 3)', 'p_stage NOT IN (1, 2)'
  );
  v_definition := pg_catalog.replace(v_definition,
    $old$OR (p_stage = 3 AND v_round.status IN ('stage3_judged', 'confirmed', 'published')) OR (p_stage = 2 AND v_round.status IN ($old$,
    $new$OR (p_stage = 2 AND v_round.status IN ($new$
  );
  v_definition := pg_catalog.replace(v_definition,
    $old$'scored', 'stage3', 'stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed', 'published'$old$,
    $new$'scored', 'published'$new$
  );
  EXECUTE v_definition;

  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(v_definition,
    'p_stage IS NULL OR p_stage NOT IN (1, 2, 3)',
    'p_stage IS NULL OR p_stage NOT IN (1, 2)'
  );
  v_definition := pg_catalog.replace(v_definition,
    $old$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN (CASE p_stage WHEN 1 THEN 0 WHEN 2 THEN 10 ELSE 20 END) AND (CASE p_stage WHEN 1 THEN 9 WHEN 2 THEN 19 ELSE 24 END)$old$,
    $new$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN (CASE p_stage WHEN 1 THEN 0 ELSE 10 END) AND (CASE p_stage WHEN 1 THEN 9 ELSE 19 END)$new$
  );
  EXECUTE v_definition;

  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(v_definition,
    $old$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring', 'stage3', 'stage3_scoring')$old$,
    $new$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring')$new$
  );
  v_definition := pg_catalog.replace(v_definition,
    $old$CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 WHEN v_round.status IN ('stage2', 'stage2_scoring') THEN 2 ELSE 3 END;
  IF v_stage = 3 AND pg_catalog.clock_timestamp() >=
       (v_round.configuration_doc #>> ARRAY['schedule',
          CASE WHEN v_round.status = 'stage3' THEN 'stage_3_close'
               ELSE 'stage_3_scoring_close' END])::TIMESTAMPTZ THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stage_closed', 'round_status', v_round.status
    );
  END IF$old$,
    $new$CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 ELSE 2 END$new$
  );
  EXECUTE v_definition;

  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_expire_leases(text)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(v_definition,
    $old$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring', 'stage3', 'stage3_scoring')$old$,
    $new$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring')$new$
  );
  EXECUTE v_definition;

  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(
    v_definition, ' WHEN 3 THEN 5::BIGINT', ''
  );
  EXECUTE v_definition;
END;
$restore_shared_stage_functions$;

DO $verify_shared_stage_functions$
DECLARE
  v_definition TEXT;
  v_signature TEXT;
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_close_stage(text,smallint)',
    'public.lab_arena_close_scoring(text,smallint)',
    'public.lab_arena_open_scoring_v2(text,smallint,jsonb)',
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
    'public.lab_arena_expire_leases(text)',
    'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(v_signature))
    INTO v_definition;
    IF v_definition IS NULL
       OR pg_catalog.strpos(v_definition, 'stage3') > 0
       OR pg_catalog.strpos(v_definition, 'stage_3_') > 0
       OR pg_catalog.strpos(v_definition, 'WHEN 3 THEN 5') > 0
       OR pg_catalog.strpos(v_definition, 'ELSE 20') > 0
       OR pg_catalog.strpos(v_definition, 'NOT IN (1, 2, 3)') > 0 THEN
      RAISE EXCEPTION 'lab_arena_two_stage_function_restore_failed: %',
        v_signature;
    END IF;
  END LOOP;
END;
$verify_shared_stage_functions$;

CREATE OR REPLACE FUNCTION public.lab_arena_record_run_scores(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_scores JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $record_scores$
DECLARE
  v_round public.lab_arena_rounds;
  v_score JSONB;
  v_run public.lab_arena_runs;
  v_recorded INTEGER := 0;
  v_existing INTEGER := 0;
  v_value NUMERIC(12, 6);
  v_qualification JSONB;
  v_integrity BOOLEAN;
BEGIN
  IF p_stage NOT IN (1, 2)
     OR pg_catalog.jsonb_typeof(p_scores) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_scores_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  v_integrity := COALESCE(
    v_round.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1',
    FALSE
  );
  IF v_round.status NOT IN (
       'stage' || p_stage::TEXT || '_closed',
       'stage' || p_stage::TEXT || '_judged'
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status
    );
  END IF;
  FOR v_score IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_scores)
  LOOP
    IF pg_catalog.jsonb_typeof(v_score) IS DISTINCT FROM 'object'
       OR COALESCE(v_score ->> 'run_id', '') = ''
       OR pg_catalog.jsonb_typeof(v_score -> 'per_icp_score')
          IS DISTINCT FROM 'number' THEN
      RAISE EXCEPTION 'lab_arena_scores_input_invalid' USING ERRCODE = '22023';
    END IF;
    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);
    IF v_value IS NULL OR v_value < -100 OR v_value > 100 THEN
      RAISE EXCEPTION 'lab_arena_score_value_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_integrity THEN
      v_qualification := v_score -> 'qualification_doc';
      IF NOT public.lab_arena__qualification_doc_valid(v_qualification) THEN
        RAISE EXCEPTION 'lab_arena_qualification_receipt_invalid'
          USING ERRCODE = '22023';
      END IF;
    ELSE
      IF v_score ? 'qualification_doc' THEN
        RAISE EXCEPTION 'lab_arena_integrity_policy_required'
          USING ERRCODE = '22023';
      END IF;
      v_qualification := NULL;
    END IF;
    SELECT * INTO v_run FROM public.lab_arena_runs
    WHERE run_id = v_score ->> 'run_id'
      AND round_id = p_round_id AND stage = p_stage
      AND kind = 'execute'
    FOR UPDATE;
    IF NOT FOUND OR v_run.status NOT IN ('accepted', 'failed') THEN
      RAISE EXCEPTION 'lab_arena_score_run_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_run.per_icp_score IS NOT NULL THEN
      IF v_run.per_icp_score = v_value
         AND v_run.qualification_doc IS NOT DISTINCT FROM v_qualification THEN
        v_existing := v_existing + 1;
        CONTINUE;
      END IF;
      RAISE EXCEPTION 'lab_arena_score_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_runs
    SET per_icp_score = v_value, qualification_doc = v_qualification
    WHERE run_id = v_run.run_id;
    v_recorded := v_recorded + 1;
  END LOOP;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'recorded', v_recorded, 'existing', v_existing
  );
END;
$record_scores$;
ALTER FUNCTION public.lab_arena_record_run_scores(TEXT, SMALLINT, JSONB)
  OWNER TO lab_arena_owner;

-- Remove the stage-three branches from the current transition RPC while
-- preserving the later open-round scorer refresh implementation.
DO $restore_transition$
DECLARE
  v_definition TEXT;
  v_plan_branch TEXT := $branch$
  ELSIF p_expected_status = 'stage3_closed' AND p_next_status = 'stage3_closed' THEN
    v_allowed := ARRAY['stage3_scoring_plan_doc'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'stage3_scoring_plan_doc') IS DISTINCT FROM 'object'
       OR v_round.configuration_doc ->> 'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage3_scoring_plan_doc IS NOT NULL THEN
      IF v_round.stage3_scoring_plan_doc = (v_patch -> 'stage3_scoring_plan_doc') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status,
          'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage3_scoring_plan_doc = v_patch -> 'stage3_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_judged' AND p_next_status = 'stage1_scored' THEN
$branch$;
  v_plan_anchor TEXT := $anchor$
  ELSIF p_expected_status = 'stage1_judged' AND p_next_status = 'stage1_scored' THEN
$anchor$;
  v_confirm_branch TEXT := $branch$
  ELSIF p_expected_status = 'stage3_judged' AND p_next_status = 'confirmed' THEN
    v_allowed := ARRAY[]::TEXT[];
    IF v_keys <> ARRAY[]::TEXT[]
       OR v_round.configuration_doc ->> 'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
       OR v_round.stage3_scoring_plan_doc IS NULL
       OR COALESCE((v_round.confirmation_cohort ->> 'required')::BOOLEAN, FALSE) IS NOT TRUE THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'confirmed', status_generation = status_generation + 1
    WHERE round_id = p_round_id;
  ELSIF ((p_expected_status = 'scored'
          AND v_round.configuration_doc ->> 'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1')
         OR (p_expected_status = 'confirmed'
          AND v_round.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1'))
        AND p_next_status = 'published' THEN
$branch$;
  v_confirm_anchor TEXT := $anchor$
  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN
$anchor$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_transition_round(text,text,text,jsonb)'
  )) INTO v_definition;
  v_definition := pg_catalog.replace(
    v_definition, v_plan_branch, v_plan_anchor
  );
  v_definition := pg_catalog.replace(
    v_definition, v_confirm_branch, v_confirm_anchor
  );
  EXECUTE v_definition;
END;
$restore_transition$;

DO $verify_transition$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(
    'public.lab_arena_transition_round(text,text,text,jsonb)'
  )) INTO v_definition;
  IF v_definition IS NULL
     OR pg_catalog.strpos(v_definition, 'stage3') > 0
     OR pg_catalog.strpos(v_definition, 'confirmed') > 0
     OR pg_catalog.strpos(v_definition,
          $expected$p_expected_status = 'scored' AND p_next_status = 'published'$expected$
        ) = 0 THEN
    RAISE EXCEPTION 'lab_arena_two_stage_transition_restore_failed';
  END IF;
END;
$verify_transition$;

-- The full integrity guard remains receipt-backed and uses exactly positions
-- 0..19. It preserves cost, qualification, omission, tie, and +1 checks.
CREATE OR REPLACE FUNCTION public.lab_arena_integrity_publication_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $publication_guard$
DECLARE
  v_baseline_id TEXT;
  v_baseline_score NUMERIC;
  v_ranking JSONB;
  v_submission_id TEXT;
  v_participant JSONB;
  v_main JSONB;
  v_eligibility JSONB;
  v_doc_eligible BOOLEAN;
  v_doc_reason TEXT;
  v_cost JSONB;
  v_qualified NUMERIC;
  v_decision JSONB;
  v_winner_id TEXT;
  v_winner_score NUMERIC;
  v_round_participant JSONB;
  v_ranked_count INTEGER;
  v_persisted_count INTEGER;
  v_selected BOOLEAN;
  v_allowed_failure BOOLEAN;
BEGIN
  IF NEW.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR NEW.status <> 'published' OR OLD.status = 'published' THEN
    RETURN NEW;
  END IF;
  IF OLD.status <> 'scored'
     OR pg_catalog.jsonb_typeof(NEW.publication_doc -> 'participants')
        IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(NEW.publication_doc -> 'final_ranking')
        IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_integrity_publication_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_array_length(NEW.publication_doc -> 'participants')
       <> pg_catalog.jsonb_array_length(NEW.participants)
     OR (SELECT pg_catalog.count(DISTINCT participant ->> 'submission_id')
         FROM pg_catalog.jsonb_array_elements(
           NEW.publication_doc -> 'participants'
         ) AS participant)
       <> pg_catalog.jsonb_array_length(NEW.participants)
     OR EXISTS (
       SELECT 1
       FROM pg_catalog.jsonb_array_elements(
         NEW.publication_doc -> 'participants'
       ) AS published
       WHERE pg_catalog.jsonb_typeof(published) IS DISTINCT FROM 'object'
          OR (SELECT pg_catalog.count(*)
              FROM pg_catalog.jsonb_object_keys(published)) <> 3
          OR NOT published ?& ARRAY[
            'submission_id', 'miner_hotkey', 'is_baseline'
          ]
          OR published ? 'is_king'
          OR pg_catalog.jsonb_typeof(published -> 'is_baseline')
             IS DISTINCT FROM 'boolean'
          OR NOT EXISTS (
            SELECT 1
            FROM pg_catalog.jsonb_array_elements(NEW.participants) AS original
            WHERE original ->> 'submission_id' = published ->> 'submission_id'
              AND original ->> 'miner_hotkey' = published ->> 'miner_hotkey'
              AND COALESCE((original ->> 'is_king')::BOOLEAN, FALSE)
                    = (published ->> 'is_baseline')::BOOLEAN
          )
     ) THEN
    RAISE EXCEPTION 'lab_arena_publication_participants_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.min(participant ->> 'submission_id'), pg_catalog.count(*)
  INTO v_baseline_id, v_qualified
  FROM pg_catalog.jsonb_array_elements(NEW.participants) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  IF v_qualified <> 1 OR COALESCE(v_baseline_id, '') = ''
     OR (SELECT pg_catalog.count(DISTINCT ranking ->> 'submission_id')
         FROM pg_catalog.jsonb_array_elements(
           NEW.publication_doc -> 'final_ranking'
         ) AS ranking)
        <> pg_catalog.jsonb_array_length(
          NEW.publication_doc -> 'final_ranking'
        ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;

  FOR v_round_participant IN
    SELECT value FROM pg_catalog.jsonb_array_elements(NEW.participants)
  LOOP
    v_submission_id := v_round_participant ->> 'submission_id';
    SELECT pg_catalog.count(*) INTO v_ranked_count
    FROM pg_catalog.jsonb_array_elements(
      NEW.publication_doc -> 'final_ranking'
    ) AS ranking
    WHERE ranking ->> 'submission_id' = v_submission_id;
    SELECT pg_catalog.count(DISTINCT runs.icp_position)
    INTO v_persisted_count
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = NEW.round_id
      AND runs.submission_id = v_submission_id
      AND runs.kind = 'execute'
      AND runs.icp_position BETWEEN 0 AND 19
      AND runs.per_icp_score IS NOT NULL;
    v_selected := COALESCE(
      (v_round_participant ->> 'is_king')::BOOLEAN, FALSE
    ) OR COALESCE(NEW.finalists, '[]'::JSONB)
      @> pg_catalog.jsonb_build_array(v_submission_id);
    IF v_persisted_count = 20 THEN
      IF v_ranked_count <> 1 THEN
        RAISE EXCEPTION 'lab_arena_publication_ranking_incomplete'
          USING ERRCODE = '22023';
      END IF;
      CONTINUE;
    END IF;
    IF v_ranked_count <> 0 THEN
      RAISE EXCEPTION 'lab_arena_publication_ranking_invalid'
        USING ERRCODE = '22023';
    END IF;
    IF COALESCE((v_round_participant ->> 'is_king')::BOOLEAN, FALSE) THEN
      RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
        USING ERRCODE = '22023';
    END IF;
    -- A stage-one nonfinalist has no stage-two rows and is intentionally absent
    -- from the final ranking. Only the baseline and frozen finalists require a
    -- complete twenty-position result or durable account-failure evidence.
    IF NOT v_selected THEN
      CONTINUE;
    END IF;
    WITH latest AS (
      SELECT DISTINCT ON (runs.assignment_id)
        runs.status, runs.terminal_cause
      FROM public.lab_arena_runs AS runs
      WHERE runs.round_id = NEW.round_id
        AND runs.submission_id = v_submission_id
        AND runs.stage IN (1, 2)
        AND runs.kind = 'score'
      ORDER BY runs.assignment_id,
        (runs.status = 'accepted') DESC, runs.attempt DESC
    )
    SELECT EXISTS (
             SELECT 1 FROM latest
             WHERE status <> 'accepted'
               AND terminal_cause IN ('credential_error', 'budget_exhausted')
           )
           AND NOT EXISTS (
             SELECT 1 FROM latest
             WHERE status <> 'accepted'
               AND (
                 terminal_cause IS NULL
                 OR terminal_cause NOT IN (
                   'credential_error', 'budget_exhausted'
                 )
               )
           )
    INTO v_allowed_failure;
    IF NOT COALESCE(v_allowed_failure, FALSE) THEN
      RAISE EXCEPTION 'lab_arena_publication_scoring_incomplete'
        USING ERRCODE = '22023';
    END IF;
  END LOOP;

  FOR v_ranking IN
    SELECT value FROM pg_catalog.jsonb_array_elements(
      NEW.publication_doc -> 'final_ranking'
    )
  LOOP
    v_submission_id := v_ranking ->> 'submission_id';
    SELECT participant INTO v_participant
    FROM pg_catalog.jsonb_array_elements(NEW.participants) AS participant
    WHERE participant ->> 'submission_id' = v_submission_id;
    IF v_participant IS NULL
       OR pg_catalog.jsonb_typeof(v_ranking -> 'is_baseline')
          IS DISTINCT FROM 'boolean'
       OR (v_ranking ->> 'is_baseline')::BOOLEAN IS DISTINCT FROM COALESCE(
            (v_participant ->> 'is_king')::BOOLEAN, FALSE
          )
       OR pg_catalog.jsonb_typeof(v_ranking -> 'eligible')
          IS DISTINCT FROM 'boolean' THEN
      RAISE EXCEPTION 'lab_arena_integrity_publication_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_main := public.lab_arena__integrity_submission_summary(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    IF NOT COALESCE((v_main ->> 'valid')::BOOLEAN, FALSE)
       OR NOT (v_ranking ? 'final_score')
       OR (v_main -> 'score' = 'null'::JSONB
           AND v_ranking -> 'final_score' <> 'null'::JSONB)
       OR (v_main -> 'score' <> 'null'::JSONB AND (
         pg_catalog.jsonb_typeof(v_ranking -> 'final_score')
           IS DISTINCT FROM 'number'
         OR (v_ranking ->> 'final_score')::NUMERIC
           IS DISTINCT FROM (v_main ->> 'score')::NUMERIC
       )) THEN
      RAISE EXCEPTION 'lab_arena_final_score_mismatch'
        USING ERRCODE = '22023';
    END IF;
    v_eligibility := public.lab_arena__integrity_eligibility(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    v_doc_eligible := (v_ranking ->> 'eligible')::BOOLEAN;
    v_doc_reason := v_ranking ->> 'eligibility_reason';
    IF v_doc_eligible IS DISTINCT FROM
         (v_eligibility ->> 'eligible')::BOOLEAN
       OR v_doc_reason IS DISTINCT FROM
          v_eligibility ->> 'eligibility_reason' THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
        USING ERRCODE = '22023';
    END IF;
    v_cost := v_ranking -> 'cost_summary';
    IF pg_catalog.jsonb_typeof(v_cost) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(v_cost -> 'returned_company_count')
          IS DISTINCT FROM 'number'
       OR (v_cost ->> 'returned_company_count')::NUMERIC < 0
       OR (v_cost ->> 'returned_company_count')::NUMERIC <>
          pg_catalog.trunc((v_cost ->> 'returned_company_count')::NUMERIC)
       OR pg_catalog.jsonb_typeof(v_cost -> 'qualified_company_count')
          IS DISTINCT FROM 'number'
       OR (v_cost ->> 'qualified_company_count')::NUMERIC < 0
       OR (v_cost ->> 'qualified_company_count')::NUMERIC > 100
       OR (v_cost ->> 'qualified_company_count')::NUMERIC <>
          pg_catalog.trunc((v_cost ->> 'qualified_company_count')::NUMERIC) THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_qualified := (v_cost ->> 'qualified_company_count')::NUMERIC;
    IF v_qualified IS DISTINCT FROM
         (v_eligibility ->> 'qualified_company_count')::NUMERIC
       OR (v_cost ->> 'execution_cap_microusd')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'execution_cap_microusd')::BIGINT
       OR (v_cost ->> 'cost_per_company_cap_microusd')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'cost_per_company_cap_microusd')::BIGINT
       OR (v_cost ->> 'eligibility_cap_microusd')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'eligibility_cap_microusd')::BIGINT
       OR (v_cost #>> '{execution,settled_microusd}')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'settled_microusd')::BIGINT
       OR (v_cost #>> '{execution,reserved_or_uncertain_microusd}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'reserved_or_uncertain_microusd')::BIGINT
       OR (v_cost #>> '{execution,conservative_microusd}')::BIGINT
          IS DISTINCT FROM (v_eligibility ->> 'conservative_microusd')::BIGINT
       OR (v_cost #>> '{execution,inflight_calls}')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'execution_inflight_calls')::BIGINT
       OR (v_cost #>> '{execution,uncertain_calls}')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'execution_uncertain_calls')::BIGINT
       OR (v_cost #>> '{judge,inflight_calls}')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'judge_inflight_calls')::BIGINT
       OR (v_cost #>> '{judge,uncertain_calls}')::BIGINT IS DISTINCT FROM
          (v_eligibility ->> 'judge_uncertain_calls')::BIGINT THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
        USING ERRCODE = '22023';
    END IF;
    IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_v1' THEN
      PERFORM public.lab_arena__successful_call_publication_valid(
        NEW.round_id, v_ranking, v_qualified::BIGINT
      );
    END IF;
    IF v_submission_id = v_baseline_id THEN
      IF v_ranking -> 'final_score' = 'null'::JSONB THEN
        RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
          USING ERRCODE = '22023';
      END IF;
      v_baseline_score := (v_ranking ->> 'final_score')::NUMERIC;
    END IF;
  END LOOP;
  IF v_baseline_score IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;

  v_decision := NEW.publication_doc -> 'king_decision';
  IF pg_catalog.jsonb_typeof(v_decision) IS DISTINCT FROM 'object'
     OR v_decision ->> 'outcome' NOT IN ('crowned', 'no_king') THEN
    RAISE EXCEPTION 'lab_arena_publication_decision_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT ranking ->> 'submission_id',
         (ranking ->> 'final_score')::NUMERIC
  INTO v_winner_id, v_winner_score
  FROM pg_catalog.jsonb_array_elements(
    NEW.publication_doc -> 'final_ranking'
  ) AS ranking
  WHERE NOT COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
    AND COALESCE((ranking ->> 'eligible')::BOOLEAN, FALSE)
    AND pg_catalog.jsonb_typeof(ranking -> 'final_score') = 'number'
    AND (ranking ->> 'final_score')::NUMERIC >= v_baseline_score + 1
  ORDER BY (ranking ->> 'final_score')::NUMERIC DESC,
           ranking ->> 'submission_id'
  LIMIT 1;
  IF v_winner_id IS NULL THEN
    IF v_decision ->> 'outcome' <> 'no_king'
       OR COALESCE(v_decision ->> 'winner_submission_id', '') <> ''
       OR COALESCE(v_decision ->> 'king_submission_id', '') <> ''
       OR COALESCE(v_decision ->> 'king_hotkey', '') <> '' THEN
      RAISE EXCEPTION 'lab_arena_publication_decision_invalid'
        USING ERRCODE = '22023';
    END IF;
  ELSE
    SELECT participant INTO v_participant
    FROM pg_catalog.jsonb_array_elements(NEW.participants) AS participant
    WHERE participant ->> 'submission_id' = v_winner_id;
    IF v_decision ->> 'outcome' <> 'crowned'
       OR v_decision ->> 'winner_submission_id' IS DISTINCT FROM v_winner_id
       OR v_decision ->> 'king_submission_id' IS DISTINCT FROM v_winner_id
       OR v_decision ->> 'king_hotkey'
          IS DISTINCT FROM v_participant ->> 'miner_hotkey' THEN
      RAISE EXCEPTION 'lab_arena_publication_winner_invalid'
        USING ERRCODE = '22023';
    END IF;
  END IF;
  RETURN NEW;
END;
$publication_guard$;
ALTER FUNCTION public.lab_arena_integrity_publication_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_integrity_publication_guard_v1()
  FROM PUBLIC;

DROP FUNCTION IF EXISTS public.lab_arena_prepare_confirmation_bank(
  TEXT, TEXT, TEXT
);
DROP FUNCTION IF EXISTS public.lab_arena_open_confirmation(TEXT, JSONB);
DROP FUNCTION IF EXISTS public.lab_arena__confirmation_account_failure(
  TEXT, TEXT
);

CREATE OR REPLACE FUNCTION public.lab_arena_twenty_icp_promotion_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version',
      'leadpoet.lab_arena.twenty_icp_promotion_schema.v1',
    'version', 248
  );
$schema$;
ALTER FUNCTION public.lab_arena_twenty_icp_promotion_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_twenty_icp_promotion_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_twenty_icp_promotion_schema_v1()
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
