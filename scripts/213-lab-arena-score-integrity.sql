-- Add the private five-ICP confirmation stage and durable qualification
-- receipts for rounds that explicitly opt into arena_integrity_v1.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $lab_arena_213_requires_212$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_register_submission_v2(text,text,text,jsonb,text,bigint,text)'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_submissions_one_active_owner_uq'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply 211-lab-arena-owner-admission.sql and 212-lab-arena-accepted-judgment-cache.sql first';
  END IF;
END;
$lab_arena_213_requires_212$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS confirmation_bank_ref TEXT;
ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS confirmation_bank_hash TEXT;
ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS confirmation_cohort JSONB;
ALTER TABLE public.lab_arena_rounds
  ADD COLUMN IF NOT EXISTS stage3_scoring_plan_doc JSONB;
ALTER TABLE public.lab_arena_runs
  ADD COLUMN IF NOT EXISTS qualification_doc JSONB;

ALTER TABLE public.lab_arena_rounds
  DROP CONSTRAINT IF EXISTS lab_arena_rounds_status_check;
ALTER TABLE public.lab_arena_rounds
  ADD CONSTRAINT lab_arena_rounds_status_check CHECK (status IN (
    'open', 'committed', 'stage1', 'stage1_closed', 'stage1_scoring',
    'stage1_judged', 'stage1_scored', 'stage2', 'stage2_closed',
    'stage2_scoring', 'stage2_judged', 'scored', 'stage3',
    'stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed',
    'published', 'cancelled'
  ));
ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_stage_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_stage_check CHECK (stage IN (1, 2, 3));
ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_icp_position_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_icp_position_check
  CHECK (icp_position BETWEEN 0 AND 29);
ALTER TABLE public.lab_arena_ledger
  DROP CONSTRAINT IF EXISTS lab_arena_ledger_stage_check;
ALTER TABLE public.lab_arena_ledger
  ADD CONSTRAINT lab_arena_ledger_stage_check
  CHECK (stage IS NULL OR stage IN (1, 2, 3));

-- Repair the all-null-or-complete tuple check if an early local 211 draft was
-- replayed before the explicit NOT NULL arms were added.
ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_owner_admission_shape;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_owner_admission_shape CHECK (
    (owner_coldkey IS NULL AND owner_block_number IS NULL
     AND owner_block_hash IS NULL)
    OR (
      owner_coldkey IS NOT NULL AND owner_block_number IS NOT NULL
      AND owner_block_hash IS NOT NULL
      AND owner_coldkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
      AND owner_block_number >= 0
      AND owner_block_hash ~ '^0x[0-9a-f]{64}$'
    )
  );

DO $lab_arena_213_round_column_constraints$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_constraint
    WHERE conrelid = 'public.lab_arena_rounds'::pg_catalog.regclass
      AND conname = 'lab_arena_rounds_confirmation_bank_shape'
  ) THEN
    ALTER TABLE public.lab_arena_rounds
      ADD CONSTRAINT lab_arena_rounds_confirmation_bank_shape CHECK (
        (confirmation_bank_ref IS NULL AND confirmation_bank_hash IS NULL)
        OR (
          confirmation_bank_ref IS NOT NULL
          AND confirmation_bank_hash IS NOT NULL
          AND pg_catalog.char_length(confirmation_bank_ref) BETWEEN 1 AND 1024
          AND confirmation_bank_hash ~ '^sha256:[0-9a-f]{64}$'
        )
      );
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_constraint
    WHERE conrelid = 'public.lab_arena_runs'::pg_catalog.regclass
      AND conname = 'lab_arena_runs_qualification_doc_shape'
  ) THEN
    ALTER TABLE public.lab_arena_runs
      ADD CONSTRAINT lab_arena_runs_qualification_doc_shape CHECK (
        qualification_doc IS NULL
        OR pg_catalog.jsonb_typeof(qualification_doc) = 'object'
      );
  END IF;
END;
$lab_arena_213_round_column_constraints$;

CREATE OR REPLACE FUNCTION public.lab_arena__qualification_doc_valid(
  p_doc JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog
AS $lab_arena_qualification_doc_valid$
DECLARE
  v_company JSONB;
  v_count INTEGER := 0;
  v_indexes INTEGER[] := ARRAY[]::INTEGER[];
  v_index NUMERIC;
BEGIN
  IF pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.count(*)
         FROM pg_catalog.jsonb_object_keys(p_doc)) <> 1
     OR pg_catalog.jsonb_typeof(p_doc -> 'companies')
        IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_array_length(p_doc -> 'companies') > 5 THEN
    RETURN FALSE;
  END IF;
  FOR v_company IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_doc -> 'companies')
  LOOP
    v_count := v_count + 1;
    IF pg_catalog.jsonb_typeof(v_company) IS DISTINCT FROM 'object'
       OR (SELECT pg_catalog.count(*)
           FROM pg_catalog.jsonb_object_keys(v_company)) <> 4
       OR NOT v_company ?& ARRAY[
         'company_index', 'company_identity_key',
         'company_qualified', 'duplicate_company'
       ]
       OR pg_catalog.jsonb_typeof(v_company -> 'company_index')
          IS DISTINCT FROM 'number'
       OR pg_catalog.jsonb_typeof(v_company -> 'company_identity_key')
          IS DISTINCT FROM 'string'
       OR pg_catalog.char_length(v_company ->> 'company_identity_key')
          NOT BETWEEN 1 AND 512
       OR pg_catalog.jsonb_typeof(v_company -> 'company_qualified')
          IS DISTINCT FROM 'boolean'
       OR pg_catalog.jsonb_typeof(v_company -> 'duplicate_company')
          IS DISTINCT FROM 'boolean' THEN
      RETURN FALSE;
    END IF;
    v_index := (v_company ->> 'company_index')::NUMERIC;
    IF v_index NOT BETWEEN 0 AND 4
       OR v_index <> pg_catalog.trunc(v_index)
       OR v_index::INTEGER = ANY(v_indexes) THEN
      RETURN FALSE;
    END IF;
    v_indexes := pg_catalog.array_append(v_indexes, v_index::INTEGER);
  END LOOP;
  RETURN v_count = pg_catalog.cardinality(v_indexes);
EXCEPTION WHEN OTHERS THEN
  RETURN FALSE;
END;
$lab_arena_qualification_doc_valid$;
ALTER FUNCTION public.lab_arena__qualification_doc_valid(JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__qualification_doc_valid(JSONB)
  FROM PUBLIC;

CREATE OR REPLACE FUNCTION public.lab_arena__integrity_submission_summary(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_positions INTEGER[]
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_integrity_submission_summary$
DECLARE
  v_expected INTEGER;
  v_count INTEGER;
  v_valid INTEGER;
  v_accepted INTEGER;
  v_score DOUBLE PRECISION;
  v_qualified BIGINT;
BEGIN
  v_expected := pg_catalog.cardinality(p_positions);
  IF v_expected IS NULL OR v_expected < 1
     OR EXISTS (
       SELECT 1 FROM pg_catalog.unnest(p_positions) AS position(value)
       WHERE value NOT BETWEEN 0 AND 24
     )
     OR (SELECT pg_catalog.count(DISTINCT value)
         FROM pg_catalog.unnest(p_positions) AS position(value)) <> v_expected THEN
    RAISE EXCEPTION 'lab_arena_integrity_positions_invalid'
      USING ERRCODE = '22023';
  END IF;
  WITH selected AS (
    SELECT DISTINCT ON (runs.icp_position)
      runs.icp_position, runs.status, runs.per_icp_score,
      runs.qualification_doc
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id
      AND runs.submission_id = p_submission_id
      AND runs.kind = 'execute'
      AND runs.icp_position = ANY (p_positions)
      AND runs.per_icp_score IS NOT NULL
      AND runs.status IN ('accepted', 'failed')
    ORDER BY runs.icp_position,
      (runs.status = 'accepted') DESC, runs.attempt DESC
  ), qualified AS (
    SELECT selected.icp_position,
      company ->> 'company_identity_key' AS identity_key
    FROM selected
    CROSS JOIN LATERAL pg_catalog.jsonb_array_elements(
      CASE WHEN public.lab_arena__qualification_doc_valid(
                  selected.qualification_doc
           ) THEN selected.qualification_doc -> 'companies'
           ELSE '[]'::JSONB END
    ) AS company
    WHERE (company ->> 'company_qualified')::BOOLEAN
      AND NOT (company ->> 'duplicate_company')::BOOLEAN
  )
  SELECT
    (SELECT pg_catalog.count(*) FROM selected),
    (SELECT pg_catalog.count(*) FROM selected
      WHERE public.lab_arena__qualification_doc_valid(qualification_doc)),
    (SELECT pg_catalog.count(*) FROM selected WHERE status = 'accepted'),
    (SELECT CASE WHEN pg_catalog.count(*) FILTER (WHERE status = 'accepted') > 0
            THEN pg_catalog.avg(per_icp_score)::DOUBLE PRECISION END
       FROM selected),
    (SELECT pg_catalog.count(*) FROM (
      SELECT DISTINCT icp_position, identity_key FROM qualified
    ) AS unique_qualified)
  INTO v_count, v_valid, v_accepted, v_score, v_qualified;
  RETURN pg_catalog.jsonb_build_object(
    'valid', v_count = v_expected AND v_valid = v_expected,
    'position_count', v_count,
    'accepted_count', v_accepted,
    'score', v_score,
    'qualified_company_count', v_qualified
  );
END;
$lab_arena_integrity_submission_summary$;
ALTER FUNCTION public.lab_arena__integrity_submission_summary(
  TEXT, TEXT, INTEGER[]
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__integrity_submission_summary(
  TEXT, TEXT, INTEGER[]
) FROM PUBLIC;

CREATE OR REPLACE FUNCTION public.lab_arena__integrity_eligibility(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_positions INTEGER[]
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_integrity_eligibility$
DECLARE
  v_round public.lab_arena_rounds;
  v_summary JSONB;
  v_execution_cap BIGINT;
  v_per_company_cap BIGINT;
  v_qualified BIGINT;
  v_settled BIGINT;
  v_reserved_or_uncertain BIGINT;
  v_inflight BIGINT;
  v_uncertain BIGINT;
  v_score_inflight BIGINT;
  v_score_uncertain BIGINT;
  v_conservative BIGINT;
  v_reason TEXT;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id;
  IF NOT FOUND OR v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1' THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required'
      USING ERRCODE = '22023';
  END IF;
  v_summary := public.lab_arena__integrity_submission_summary(
    p_round_id, p_submission_id, p_positions
  );
  v_qualified := COALESCE(
    (v_summary ->> 'qualified_company_count')::BIGINT, 0
  );
  v_execution_cap :=
    (v_round.configuration_doc ->> 'execution_cap_microusd')::BIGINT;
  v_per_company_cap :=
    (v_round.configuration_doc ->> 'cost_per_company_microusd')::BIGINT;
  IF v_execution_cap IS NULL OR v_execution_cap < 1
     OR v_per_company_cap IS NULL OR v_per_company_cap < 1 THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT
    COALESCE(SUM(head.amount_microusd)
      FILTER (WHERE head.entry_kind = 'settlement'), 0)::BIGINT,
    COALESCE(SUM(head.amount_microusd)
      FILTER (WHERE head.entry_kind IN
        ('reservation', 'dispatch', 'uncertain')), 0)::BIGINT,
    COUNT(*) FILTER (WHERE head.entry_kind IN
      ('reservation', 'dispatch'))::BIGINT,
    COUNT(*) FILTER (WHERE head.entry_kind = 'uncertain')::BIGINT
  INTO v_settled, v_reserved_or_uncertain, v_inflight, v_uncertain
  FROM (
    SELECT DISTINCT ON (ledger.call_identity)
      ledger.entry_kind, ledger.amount_microusd
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND runs.kind = 'execute'
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head;
  SELECT
    COUNT(*) FILTER (WHERE head.entry_kind IN
      ('reservation', 'dispatch'))::BIGINT,
    COUNT(*) FILTER (WHERE head.entry_kind = 'uncertain')::BIGINT
  INTO v_score_inflight, v_score_uncertain
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND runs.kind = 'score'
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head;
  v_conservative := v_settled + v_reserved_or_uncertain;
  IF NOT COALESCE((v_summary ->> 'valid')::BOOLEAN, FALSE)
     OR v_summary -> 'score' = 'null'::JSONB THEN
    v_reason := 'stored_output_invalid';
  ELSIF v_inflight > 0 OR v_score_inflight > 0 THEN
    v_reason := 'provider_calls_inflight';
  ELSIF v_uncertain > 0 OR v_score_uncertain > 0 THEN
    v_reason := 'provider_cost_uncertain';
  ELSIF v_conservative > v_execution_cap THEN
    v_reason := 'execution_cap_exceeded';
  ELSIF v_conservative > v_per_company_cap * v_qualified THEN
    v_reason := 'cost_per_company_exceeded';
  ELSE
    v_reason := 'eligible';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'eligible', v_reason = 'eligible',
    'eligibility_reason', v_reason,
    'qualified_company_count', v_qualified,
    'execution_cap_microusd', v_execution_cap,
    'cost_per_company_cap_microusd', v_per_company_cap,
    'eligibility_cap_microusd', LEAST(
      v_execution_cap, v_per_company_cap * v_qualified
    ),
    'settled_microusd', v_settled,
    'reserved_or_uncertain_microusd', v_reserved_or_uncertain,
    'conservative_microusd', v_conservative,
    'execution_inflight_calls', v_inflight,
    'execution_uncertain_calls', v_uncertain,
    'judge_inflight_calls', v_score_inflight,
    'judge_uncertain_calls', v_score_uncertain
  );
END;
$lab_arena_integrity_eligibility$;
ALTER FUNCTION public.lab_arena__integrity_eligibility(
  TEXT, TEXT, INTEGER[]
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__integrity_eligibility(
  TEXT, TEXT, INTEGER[]
) FROM PUBLIC;

-- The private confirmation bank is committed before the public benchmark is
-- committed.  Its location and hash are write-once, and old rounds cannot
-- acquire confirmation state after the fact.
CREATE OR REPLACE FUNCTION public.lab_arena_prepare_confirmation_bank(
  p_round_id TEXT,
  p_ref TEXT,
  p_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_prepare_confirmation_bank$
DECLARE
  v_round public.lab_arena_rounds;
BEGIN
  IF pg_catalog.char_length(COALESCE(p_ref, '')) NOT BETWEEN 1 AND 1024
     OR COALESCE(p_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR p_ref IS DISTINCT FROM 'arena/' || p_round_id || '/confirmation/'
        || pg_catalog.substr(p_hash, 8) || '.json' THEN
    RAISE EXCEPTION 'lab_arena_confirmation_bank_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1' THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required'
      USING ERRCODE = '22023';
  END IF;
  IF v_round.status <> 'open' THEN
    IF v_round.confirmation_bank_ref IS NOT NULL
       AND v_round.confirmation_bank_ref = p_ref
       AND v_round.confirmation_bank_hash = p_hash THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status
    );
  END IF;
  IF v_round.confirmation_bank_ref IS NOT NULL THEN
    IF v_round.confirmation_bank_ref = p_ref
       AND v_round.confirmation_bank_hash = p_hash THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing',
        'round_status', v_round.status);
    END IF;
    RAISE EXCEPTION 'lab_arena_confirmation_bank_write_once'
      USING ERRCODE = '42501';
  END IF;
  UPDATE public.lab_arena_rounds
  SET confirmation_bank_ref = p_ref, confirmation_bank_hash = p_hash
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_round.status
  );
END;
$lab_arena_prepare_confirmation_bank$;
ALTER FUNCTION public.lab_arena_prepare_confirmation_bank(TEXT, TEXT, TEXT)
  OWNER TO lab_arena_owner;

-- Enforce integrity-only columns and stage ranges even against direct table
-- writes by the owning role.  Existing round/run triggers retain their older
-- mutation rules; this trigger only adds the new policy invariants.
CREATE OR REPLACE FUNCTION public.lab_arena_integrity_round_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog
AS $lab_arena_integrity_round_guard$
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
    NEW.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1',
    FALSE
  );
  IF NOT v_integrity AND (
       NEW.confirmation_bank_ref IS NOT NULL
       OR NEW.confirmation_bank_hash IS NOT NULL
       OR NEW.confirmation_cohort IS NOT NULL
       OR NEW.stage3_scoring_plan_doc IS NOT NULL
       OR NEW.status IN ('stage3', 'stage3_closed', 'stage3_scoring',
                         'stage3_judged', 'confirmed')
     ) THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required'
      USING ERRCODE = '23514';
  END IF;
  IF v_integrity THEN
    IF NEW.configuration_doc #>> '{scorer_policy,scoring_adapter_version}'
         IS DISTINCT FROM 'qualification_integrity_v2'
       OR NOT (NEW.configuration_doc ? 'cost_per_company_microusd') THEN
      RAISE EXCEPTION 'lab_arena_integrity_configuration_invalid'
        USING ERRCODE = '23514';
    END IF;
    IF NEW.status NOT IN ('open', 'cancelled')
       AND (NEW.confirmation_bank_ref IS NULL
            OR NEW.confirmation_bank_hash IS NULL) THEN
      RAISE EXCEPTION 'lab_arena_confirmation_bank_required'
        USING ERRCODE = '23514';
    END IF;
    IF NEW.status IN ('stage3', 'stage3_closed', 'stage3_scoring',
                      'stage3_judged', 'confirmed', 'published')
       AND NEW.confirmation_cohort IS NULL THEN
      RAISE EXCEPTION 'lab_arena_confirmation_cohort_required'
        USING ERRCODE = '23514';
    END IF;
    IF NEW.status = 'published' AND TG_OP = 'UPDATE'
       AND OLD.status <> 'published' AND OLD.status <> 'confirmed' THEN
      RAISE EXCEPTION 'lab_arena_confirmation_required'
        USING ERRCODE = '23514';
    END IF;
  END IF;
  IF TG_OP = 'UPDATE' AND (
       (OLD.confirmation_bank_ref IS NOT NULL
        AND NEW.confirmation_bank_ref IS DISTINCT FROM OLD.confirmation_bank_ref)
       OR (OLD.confirmation_bank_hash IS NOT NULL
        AND NEW.confirmation_bank_hash IS DISTINCT FROM OLD.confirmation_bank_hash)
       OR (OLD.confirmation_cohort IS NOT NULL
        AND NEW.confirmation_cohort IS DISTINCT FROM OLD.confirmation_cohort)
       OR (OLD.stage3_scoring_plan_doc IS NOT NULL
        AND NEW.stage3_scoring_plan_doc IS DISTINCT FROM OLD.stage3_scoring_plan_doc)
     ) THEN
    RAISE EXCEPTION 'lab_arena_integrity_state_write_once'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$lab_arena_integrity_round_guard$;
ALTER FUNCTION public.lab_arena_integrity_round_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_integrity_round_guard_v1()
  FROM PUBLIC;

DROP TRIGGER IF EXISTS lab_arena_integrity_round_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_integrity_round_guard
  BEFORE INSERT OR UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_integrity_round_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_integrity_run_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_integrity_run_guard$
DECLARE
  v_round public.lab_arena_rounds;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = NEW.round_id;
  IF NOT FOUND THEN RETURN NEW; END IF;
  IF (NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND 9)
     OR (NEW.stage = 2
         AND NEW.icp_position NOT BETWEEN 10 AND
           CASE WHEN v_round.configuration_doc ->> 'integrity_policy'
                       = 'arena_integrity_v1' THEN 19 ELSE 29 END)
     OR (NEW.stage = 3 AND NEW.icp_position NOT BETWEEN 20 AND 24) THEN
    RAISE EXCEPTION 'lab_arena_run_position_invalid'
      USING ERRCODE = '23514';
  END IF;
  IF NEW.stage = 3 THEN
    IF v_round.configuration_doc ->> 'integrity_policy'
         IS DISTINCT FROM 'arena_integrity_v1'
       OR COALESCE((v_round.confirmation_cohort ->> 'required')::BOOLEAN,
                   FALSE) IS NOT TRUE
       OR NOT (v_round.confirmation_cohort -> 'submission_ids'
               @> pg_catalog.jsonb_build_array(NEW.submission_id)) THEN
      RAISE EXCEPTION 'lab_arena_confirmation_run_invalid'
        USING ERRCODE = '23514';
    END IF;
  END IF;
  IF v_round.configuration_doc ->> 'integrity_policy'
       = 'arena_integrity_v1' THEN
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
  IF TG_OP = 'UPDATE' AND OLD.qualification_doc IS NOT NULL
     AND NEW.qualification_doc IS DISTINCT FROM OLD.qualification_doc THEN
    RAISE EXCEPTION 'lab_arena_qualification_receipt_write_once'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$lab_arena_integrity_run_guard$;
ALTER FUNCTION public.lab_arena_integrity_run_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_integrity_run_guard_v1()
  FROM PUBLIC;

DROP TRIGGER IF EXISTS lab_arena_integrity_run_guard ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_integrity_run_guard
  BEFORE INSERT OR UPDATE ON public.lab_arena_runs
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_integrity_run_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_open_confirmation(
  p_round_id TEXT,
  p_cohort JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_open_confirmation$
DECLARE
  v_round public.lab_arena_rounds;
  v_baseline_id TEXT;
  v_baseline_score NUMERIC;
  v_ids TEXT[];
  v_expected_ids TEXT[];
  v_id TEXT;
  v_entry JSONB;
  v_participant JSONB;
  v_summary JSONB;
  v_eligibility JSONB;
  v_generation BIGINT;
  v_position INTEGER;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_required BOOLEAN;
BEGIN
  IF pg_catalog.jsonb_typeof(p_cohort) IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_cohort)) <> 6
     OR NOT p_cohort ?& ARRAY['schema_version', 'submission_ids',
       'baseline_submission_id', 'main_entries', 'main_eligibility', 'required']
     OR p_cohort ->> 'schema_version'
       IS DISTINCT FROM 'leadpoet.lab_arena.confirmation_cohort.v1'
     OR pg_catalog.jsonb_typeof(p_cohort -> 'submission_ids')
       IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(p_cohort -> 'main_entries')
       IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(p_cohort -> 'main_eligibility')
       IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(p_cohort -> 'required')
       IS DISTINCT FROM 'boolean' THEN
    RAISE EXCEPTION 'lab_arena_confirmation_cohort_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT COALESCE(pg_catalog.array_agg(value ORDER BY ordinal), ARRAY[]::TEXT[])
  INTO v_ids
  FROM pg_catalog.jsonb_array_elements_text(p_cohort -> 'submission_ids')
       WITH ORDINALITY AS item(value, ordinal);
  v_required := (p_cohort ->> 'required')::BOOLEAN;
  IF pg_catalog.cardinality(v_ids) NOT BETWEEN 1 AND 4
     OR pg_catalog.cardinality(v_ids) <>
        (SELECT pg_catalog.count(DISTINCT value)
         FROM pg_catalog.unnest(v_ids) AS item(value))
     OR pg_catalog.jsonb_array_length(p_cohort -> 'main_entries')
          <> pg_catalog.cardinality(v_ids)
     OR (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(
          p_cohort -> 'main_eligibility')) <> pg_catalog.cardinality(v_ids)
     OR v_required IS DISTINCT FROM (pg_catalog.cardinality(v_ids) > 1) THEN
    RAISE EXCEPTION 'lab_arena_confirmation_cohort_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.confirmation_bank_ref IS NULL
     OR v_round.confirmation_bank_hash IS NULL THEN
    RAISE EXCEPTION 'lab_arena_confirmation_policy_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF v_round.status <> 'scored' THEN
    IF v_round.confirmation_cohort = p_cohort
       AND v_round.status IN ('stage3', 'stage3_closed', 'stage3_scoring',
                              'stage3_judged', 'confirmed', 'published') THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  IF v_round.confirmation_cohort IS NOT NULL THEN
    IF v_round.confirmation_cohort = p_cohort THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RAISE EXCEPTION 'lab_arena_confirmation_cohort_write_once'
      USING ERRCODE = '42501';
  END IF;

  SELECT participant ->> 'submission_id' INTO v_baseline_id
  FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  IF v_baseline_id IS NULL OR (
      SELECT pg_catalog.count(*)
      FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
      WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
    ) <> 1
    OR p_cohort ->> 'baseline_submission_id' IS DISTINCT FROM v_baseline_id
    OR v_ids[1] IS DISTINCT FROM v_baseline_id THEN
    RAISE EXCEPTION 'lab_arena_confirmation_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_summary := public.lab_arena__integrity_submission_summary(
    p_round_id, v_baseline_id,
    ARRAY(SELECT pg_catalog.generate_series(0, 19))
  );
  IF NOT COALESCE((v_summary ->> 'valid')::BOOLEAN, FALSE)
     OR v_summary ->> 'score' IS NULL THEN
    RAISE EXCEPTION 'lab_arena_confirmation_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_baseline_score := (v_summary ->> 'score')::NUMERIC;

  WITH candidates AS (
    SELECT participant ->> 'submission_id' AS submission_id,
      (summary.doc ->> 'score')::NUMERIC AS score
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    CROSS JOIN LATERAL (
      SELECT public.lab_arena__integrity_submission_summary(
        p_round_id, participant ->> 'submission_id',
        ARRAY(SELECT pg_catalog.generate_series(0, 19))
      ) AS doc
    ) AS summary
    CROSS JOIN LATERAL (
      SELECT public.lab_arena__integrity_eligibility(
        p_round_id, participant ->> 'submission_id',
        ARRAY(SELECT pg_catalog.generate_series(0, 19))
      ) AS doc
    ) AS eligibility
    WHERE NOT COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
      AND COALESCE((summary.doc ->> 'valid')::BOOLEAN, FALSE)
      AND COALESCE((eligibility.doc ->> 'eligible')::BOOLEAN, FALSE)
      AND (summary.doc ->> 'score')::NUMERIC >= v_baseline_score + 1
    ORDER BY score DESC, submission_id
    LIMIT 3
  )
  SELECT ARRAY[v_baseline_id] || COALESCE(
    pg_catalog.array_agg(submission_id ORDER BY score DESC, submission_id),
    ARRAY[]::TEXT[]
  ) INTO v_expected_ids FROM candidates;
  IF v_ids IS DISTINCT FROM v_expected_ids THEN
    RAISE EXCEPTION 'lab_arena_confirmation_cohort_mismatch'
      USING ERRCODE = '22023';
  END IF;

  FOR v_position IN 1 .. pg_catalog.cardinality(v_ids) LOOP
    v_id := v_ids[v_position];
    v_entry := p_cohort -> 'main_entries' -> (v_position - 1);
    SELECT participant INTO v_participant
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE participant ->> 'submission_id' = v_id;
    v_summary := public.lab_arena__integrity_submission_summary(
      p_round_id, v_id, ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    v_eligibility := public.lab_arena__integrity_eligibility(
      p_round_id, v_id, ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    IF pg_catalog.jsonb_typeof(v_entry) IS DISTINCT FROM 'object'
       OR v_entry ->> 'submission_id' IS DISTINCT FROM v_id
       OR v_entry ->> 'hotkey' IS DISTINCT FROM
          v_participant ->> 'miner_hotkey'
       OR COALESCE((v_entry ->> 'is_king')::BOOLEAN, FALSE)
          IS DISTINCT FROM COALESCE(
            (v_participant ->> 'is_king')::BOOLEAN, FALSE)
       OR pg_catalog.jsonb_typeof(v_entry -> 'final_score')
          IS DISTINCT FROM 'number'
       OR (v_entry ->> 'final_score')::NUMERIC
          IS DISTINCT FROM (v_summary ->> 'score')::NUMERIC
       OR NOT (p_cohort -> 'main_eligibility' ? v_id)
       OR pg_catalog.jsonb_typeof(
          p_cohort -> 'main_eligibility' -> v_id -> 'eligible')
          IS DISTINCT FROM 'boolean'
       OR (p_cohort #>> ARRAY['main_eligibility', v_id, 'eligible'])::BOOLEAN
          IS DISTINCT FROM (v_eligibility ->> 'eligible')::BOOLEAN
       OR p_cohort #>> ARRAY['main_eligibility', v_id, 'eligibility_reason']
          IS DISTINCT FROM v_eligibility ->> 'eligibility_reason' THEN
      RAISE EXCEPTION 'lab_arena_confirmation_evidence_mismatch'
        USING ERRCODE = '22023';
    END IF;
  END LOOP;

  v_generation := v_round.stage_generation + 1;
  IF NOT v_required THEN
    UPDATE public.lab_arena_rounds
    SET confirmation_cohort = p_cohort, status = 'confirmed',
        status_generation = status_generation + 1,
        stage_generation = v_generation
    WHERE round_id = p_round_id;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'ok', 'round_status', 'confirmed',
      'stage_generation', v_generation, 'assignments', 0
    );
  END IF;
  UPDATE public.lab_arena_rounds
  SET confirmation_cohort = p_cohort
  WHERE round_id = p_round_id;
  FOREACH v_id IN ARRAY v_ids LOOP
    SELECT participant INTO v_participant
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE participant ->> 'submission_id' = v_id;
    IF NOT EXISTS (
      SELECT 1 FROM public.lab_arena_submissions
      WHERE round_id = p_round_id AND submission_id = v_id
        AND status = 'frozen'
        AND miner_hotkey = v_participant ->> 'miner_hotkey'
    ) THEN
      RAISE EXCEPTION 'lab_arena_participant_not_frozen'
        USING ERRCODE = '23503';
    END IF;
    FOR v_position IN 20 .. 24 LOOP
      v_assignment := p_round_id || ':' || v_id || ':3:' || v_position::TEXT;
      INSERT INTO public.lab_arena_runs (
        run_id, assignment_id, round_id, submission_id, miner_hotkey,
        stage, icp_position, attempt, status, stage_generation
      ) VALUES (
        v_assignment || ':1', v_assignment, p_round_id, v_id,
        v_participant ->> 'miner_hotkey', 3, v_position, 1, 'pending',
        v_generation
      );
      v_created := v_created + 1;
    END LOOP;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET confirmation_cohort = p_cohort, status = 'stage3',
      status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', 'stage3',
    'stage_generation', v_generation, 'assignments', v_created
  );
END;
$lab_arena_open_confirmation$;
ALTER FUNCTION public.lab_arena_open_confirmation(TEXT, JSONB)
  OWNER TO lab_arena_owner;

DO $lab_arena_213_close_stage$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure('public.lab_arena_close_stage(text,smallint)')
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'p_stage NOT IN (1, 2, 3)') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'p_stage NOT IN (1, 2)') = 0
     OR pg_catalog.strpos(v_definition,
       $anchor$OR (p_stage = 2 AND v_round.status IN ($anchor$) = 0 THEN
    RAISE EXCEPTION 'lab_arena_close_stage_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    'p_stage NOT IN (1, 2)', 'p_stage NOT IN (1, 2, 3)');
  v_definition := pg_catalog.replace(v_definition,
    $anchor$OR (p_stage = 2 AND v_round.status IN ($anchor$,
    $anchor$OR (p_stage = 3 AND v_round.status IN ('stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed', 'published')) OR (p_stage = 2 AND v_round.status IN ($anchor$);
  v_definition := pg_catalog.replace(v_definition,
    $anchor$'scored', 'published'$anchor$,
    $anchor$'scored', 'stage3', 'stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed', 'published'$anchor$);
  EXECUTE v_definition;
END;
$lab_arena_213_close_stage$;

DO $lab_arena_213_close_scoring$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure('public.lab_arena_close_scoring(text,smallint)')
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'p_stage NOT IN (1, 2, 3)') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'p_stage NOT IN (1, 2)') = 0
     OR pg_catalog.strpos(v_definition,
       $anchor$OR (p_stage = 2 AND v_round.status IN ($anchor$) = 0 THEN
    RAISE EXCEPTION 'lab_arena_close_scoring_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    'p_stage NOT IN (1, 2)', 'p_stage NOT IN (1, 2, 3)');
  v_definition := pg_catalog.replace(v_definition,
    $anchor$OR (p_stage = 2 AND v_round.status IN ($anchor$,
    $anchor$OR (p_stage = 3 AND v_round.status IN ('stage3_judged', 'confirmed', 'published')) OR (p_stage = 2 AND v_round.status IN ($anchor$);
  v_definition := pg_catalog.replace(v_definition,
    $anchor$'scored', 'published'$anchor$,
    $anchor$'scored', 'stage3', 'stage3_closed', 'stage3_scoring', 'stage3_judged', 'confirmed', 'published'$anchor$);
  EXECUTE v_definition;
END;
$lab_arena_213_close_scoring$;

DO $lab_arena_213_open_scoring$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       'p_stage IS NULL OR p_stage NOT IN (1, 2, 3)') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, 'p_stage IS NULL OR p_stage < 1') = 0
     OR pg_catalog.strpos(v_definition,
       $anchor$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 29$anchor$) = 0 THEN
    RAISE EXCEPTION 'lab_arena_open_scoring_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    'p_stage IS NULL OR p_stage < 1',
    'p_stage IS NULL OR p_stage NOT IN (1, 2, 3)');
  v_definition := pg_catalog.replace(v_definition,
    $anchor$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 29$anchor$,
    $anchor$(v_item ->> 'icp_position')::INTEGER NOT BETWEEN (CASE p_stage WHEN 1 THEN 0 WHEN 2 THEN 10 ELSE 20 END) AND (CASE p_stage WHEN 1 THEN 9 WHEN 2 THEN 19 ELSE 24 END)$anchor$);
  EXECUTE v_definition;
END;
$lab_arena_213_open_scoring$;

DO $lab_arena_213_claim$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'stage3_scoring') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition,
       $anchor$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring')$anchor$) = 0
     OR pg_catalog.strpos(v_definition,
       $anchor$CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 ELSE 2 END$anchor$) = 0
     OR pg_catalog.strpos(v_definition, 'judgment_group_miner_hotkeys') = 0 THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $anchor$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring')$anchor$,
    $anchor$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring', 'stage3', 'stage3_scoring')$anchor$);
  v_definition := pg_catalog.replace(v_definition,
    $anchor$CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 ELSE 2 END$anchor$,
    $anchor$CASE WHEN v_round.status IN ('stage1', 'stage1_scoring') THEN 1 WHEN v_round.status IN ('stage2', 'stage2_scoring') THEN 2 ELSE 3 END;
  IF v_stage = 3 AND pg_catalog.clock_timestamp() >=
       (v_round.configuration_doc #>> ARRAY['schedule',
          CASE WHEN v_round.status = 'stage3' THEN 'stage_3_close'
               ELSE 'stage_3_scoring_close' END])::TIMESTAMPTZ THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stage_closed', 'round_status', v_round.status
    );
  END IF$anchor$);
  EXECUTE v_definition;
END;
$lab_arena_213_claim$;

DO $lab_arena_213_expire$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure('public.lab_arena_expire_leases(text)')
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'stage3_scoring') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition,
       $anchor$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring')$anchor$) = 0
     OR pg_catalog.strpos(v_definition, 'judgment_group_miner_hotkeys') = 0 THEN
    RAISE EXCEPTION 'lab_arena_expire_leases_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,
    $anchor$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring')$anchor$,
    $anchor$('stage1', 'stage1_scoring', 'stage2', 'stage2_scoring', 'stage3', 'stage3_scoring')$anchor$);
END;
$lab_arena_213_expire$;

DO $lab_arena_213_reserve_quota$
DECLARE v_definition TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'WHEN 3 THEN 5') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition,
       $anchor$WHEN 1 THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::BIGINT$anchor$) = 0 THEN
    RAISE EXCEPTION 'lab_arena_reserve_call_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $anchor$WHEN 1 THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::BIGINT$anchor$,
    $anchor$WHEN 1 THEN (v_round.configuration_doc ->> 'stage_1_icp_count')::BIGINT WHEN 3 THEN 5::BIGINT$anchor$);
  EXECUTE v_definition;
END;
$lab_arena_213_reserve_quota$;

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
AS $lab_arena_record_run_scores$
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
  IF p_stage NOT IN (1, 2, 3)
     OR pg_catalog.jsonb_typeof(p_scores) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_scores_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  v_integrity := COALESCE(
    v_round.configuration_doc ->> 'integrity_policy'
      = 'arena_integrity_v1',
    FALSE
  );
  IF p_stage = 3 AND NOT v_integrity THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required'
      USING ERRCODE = '22023';
  END IF;
  IF v_round.status NOT IN ('stage' || p_stage::TEXT || '_closed',
                            'stage' || p_stage::TEXT || '_judged') THEN
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
      RAISE EXCEPTION 'lab_arena_scores_input_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_value := (v_score ->> 'per_icp_score')::NUMERIC(12, 6);
    IF v_value IS NULL OR v_value < -100 OR v_value > 100 THEN
      RAISE EXCEPTION 'lab_arena_score_value_invalid'
        USING ERRCODE = '22023';
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
      RAISE EXCEPTION 'lab_arena_score_run_invalid'
        USING ERRCODE = '22023';
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
$lab_arena_record_run_scores$;
ALTER FUNCTION public.lab_arena_record_run_scores(TEXT, SMALLINT, JSONB)
  OWNER TO lab_arena_owner;

DO $lab_arena_213_transition$
DECLARE
  v_definition TEXT;
  v_plan_anchor TEXT := $anchor$
  ELSIF p_expected_status = 'stage1_judged' AND p_next_status = 'stage1_scored' THEN
$anchor$;
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
  v_confirm_anchor TEXT := $anchor$
  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN
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
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_transition_round(text,text,text,jsonb)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       $anchor$p_expected_status = 'stage3_judged'$anchor$) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_plan_anchor) = 0
     OR pg_catalog.strpos(v_definition, v_confirm_anchor) = 0 THEN
    RAISE EXCEPTION 'lab_arena_transition_round_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_plan_anchor, v_plan_branch);
  v_definition := pg_catalog.replace(v_definition, v_confirm_anchor, v_confirm_branch);
  EXECUTE v_definition;
END;
$lab_arena_213_transition$;

-- The legacy publication trigger uses raw returned-company counts and a
-- historical 100-company ceiling.  Leave that code byte-for-byte effective
-- for old rounds and route only explicitly marked rounds to the receipt-backed
-- guard below.
DO $lab_arena_213_legacy_publication_bypass$
DECLARE
  v_definition TEXT;
  v_anchor TEXT := $anchor$
BEGIN
  IF NEW.status <> 'published' OR OLD.status = 'published' THEN
$anchor$;
  v_replacement TEXT := $replacement$
BEGIN
  IF NEW.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1' THEN
    RETURN NEW;
  END IF;
  IF NEW.status <> 'published' OR OLD.status = 'published' THEN
$replacement$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_publication_baseline_guard_v1()'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition,
       $anchor$NEW.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1'$anchor$) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_anchor) = 0 THEN
    RAISE EXCEPTION 'lab_arena_publication_guard_shape_unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_anchor, v_replacement);
END;
$lab_arena_213_legacy_publication_bypass$;

CREATE OR REPLACE FUNCTION public.lab_arena_integrity_publication_guard_v1()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_integrity_publication_guard$
DECLARE
  v_baseline_id TEXT;
  v_baseline_score NUMERIC;
  v_cohort JSONB;
  v_required BOOLEAN;
  v_ranking JSONB;
  v_submission_id TEXT;
  v_participant JSONB;
  v_main JSONB;
  v_final JSONB;
  v_eligibility JSONB;
  v_positions INTEGER[];
  v_selected BOOLEAN;
  v_doc_eligible BOOLEAN;
  v_doc_reason TEXT;
  v_cost JSONB;
  v_returned NUMERIC;
  v_qualified NUMERIC;
  v_decision JSONB;
  v_winner_id TEXT;
  v_winner_score NUMERIC;
  v_round_participant JSONB;
  v_ranked_count INTEGER;
  v_persisted_count INTEGER;
  v_allowed_failure BOOLEAN;
BEGIN
  IF NEW.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR NEW.status <> 'published' OR OLD.status = 'published' THEN
    RETURN NEW;
  END IF;
  IF OLD.status <> 'confirmed'
     OR NEW.confirmation_bank_ref IS NULL
     OR NEW.confirmation_bank_hash IS NULL
     OR pg_catalog.jsonb_typeof(NEW.confirmation_cohort)
        IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(NEW.publication_doc -> 'participants')
        IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(NEW.publication_doc -> 'final_ranking')
        IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_integrity_publication_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_cohort := NEW.confirmation_cohort;
  v_required := COALESCE((v_cohort ->> 'required')::BOOLEAN, FALSE);
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
            WHERE original ->> 'submission_id'
                    = published ->> 'submission_id'
              AND original ->> 'miner_hotkey'
                    = published ->> 'miner_hotkey'
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
     OR v_cohort ->> 'baseline_submission_id' IS DISTINCT FROM v_baseline_id
     OR (v_cohort -> 'submission_ids' ->> 0) IS DISTINCT FROM v_baseline_id
     OR (SELECT pg_catalog.count(DISTINCT ranking ->> 'submission_id')
         FROM pg_catalog.jsonb_array_elements(
           NEW.publication_doc -> 'final_ranking') AS ranking)
        <> pg_catalog.jsonb_array_length(
          NEW.publication_doc -> 'final_ranking'
        ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;

  -- Final ranking is the baseline plus challengers that obtained all twenty
  -- durable main-stage scores.  A finalist may be absent only when the latest
  -- score attempt proves a miner-account failure; shared judge failures still
  -- make publication impossible.  Every frozen confirmation member is always
  -- required because its cohort was selected from complete main scores.
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
    v_selected := v_cohort -> 'submission_ids'
      @> pg_catalog.jsonb_build_array(v_submission_id);
    IF v_persisted_count = 20 OR v_selected THEN
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
    -- This branch is reached only for an omitted challenger without all twenty
    -- main scores.  Its omission needs durable miner-account failure evidence.
    IF v_persisted_count <> 20 AND NOT v_selected THEN
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
                 AND terminal_cause IN (
                   'credential_error', 'budget_exhausted'
                 )
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
       OR (v_ranking ->> 'is_baseline')::BOOLEAN
          IS DISTINCT FROM COALESCE(
            (v_participant ->> 'is_king')::BOOLEAN, FALSE)
       OR pg_catalog.jsonb_typeof(v_ranking -> 'confirmation_selected')
          IS DISTINCT FROM 'boolean'
       OR pg_catalog.jsonb_typeof(v_ranking -> 'eligible')
          IS DISTINCT FROM 'boolean' THEN
      RAISE EXCEPTION 'lab_arena_integrity_publication_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_selected := v_cohort -> 'submission_ids'
      @> pg_catalog.jsonb_build_array(v_submission_id);
    IF (v_ranking ->> 'confirmation_selected')::BOOLEAN
         IS DISTINCT FROM v_selected THEN
      RAISE EXCEPTION 'lab_arena_confirmation_selection_mismatch'
        USING ERRCODE = '22023';
    END IF;
    v_main := public.lab_arena__integrity_submission_summary(
      NEW.round_id, v_submission_id,
      ARRAY(SELECT pg_catalog.generate_series(0, 19))
    );
    IF NOT COALESCE((v_main ->> 'valid')::BOOLEAN, FALSE)
       OR NOT (v_ranking ? 'main_score')
       OR (v_main -> 'score' = 'null'::JSONB
           AND v_ranking -> 'main_score' <> 'null'::JSONB)
       OR (v_main -> 'score' <> 'null'::JSONB AND (
         pg_catalog.jsonb_typeof(v_ranking -> 'main_score')
           IS DISTINCT FROM 'number'
         OR (v_ranking ->> 'main_score')::NUMERIC
           IS DISTINCT FROM (v_main ->> 'score')::NUMERIC
       )) THEN
      RAISE EXCEPTION 'lab_arena_main_score_mismatch'
        USING ERRCODE = '22023';
    END IF;
    IF v_required AND v_selected THEN
      v_positions := ARRAY(SELECT pg_catalog.generate_series(0, 24));
      v_final := public.lab_arena__integrity_submission_summary(
        NEW.round_id, v_submission_id,
        ARRAY(SELECT pg_catalog.generate_series(20, 24))
      );
      IF NOT COALESCE((v_final ->> 'valid')::BOOLEAN, FALSE)
         OR NOT (v_ranking ? 'final_score')
         OR (v_final -> 'score' = 'null'::JSONB
             AND v_ranking -> 'final_score' <> 'null'::JSONB)
         OR (v_final -> 'score' <> 'null'::JSONB AND (
           pg_catalog.jsonb_typeof(v_ranking -> 'final_score')
             IS DISTINCT FROM 'number'
           OR (v_ranking ->> 'final_score')::NUMERIC
             IS DISTINCT FROM (v_final ->> 'score')::NUMERIC
         )) THEN
        RAISE EXCEPTION 'lab_arena_confirmation_score_mismatch'
          USING ERRCODE = '22023';
      END IF;
    ELSE
      v_positions := ARRAY(SELECT pg_catalog.generate_series(0, 19));
      IF v_required AND NOT v_selected THEN
        IF v_ranking -> 'final_score' <> 'null'::JSONB THEN
          RAISE EXCEPTION 'lab_arena_unselected_confirmation_score_invalid'
            USING ERRCODE = '22023';
        END IF;
      ELSIF NOT (v_ranking ? 'final_score')
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
    END IF;
    v_eligibility := public.lab_arena__integrity_eligibility(
      NEW.round_id, v_submission_id, v_positions
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
       OR (v_cost ->> 'qualified_company_count')::NUMERIC >
          (CASE WHEN v_selected AND v_required THEN 125 ELSE 100 END)
       OR (v_cost ->> 'qualified_company_count')::NUMERIC <>
          pg_catalog.trunc((v_cost ->> 'qualified_company_count')::NUMERIC) THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_invalid'
        USING ERRCODE = '22023';
    END IF;
    v_returned := (v_cost ->> 'returned_company_count')::NUMERIC;
    v_qualified := (v_cost ->> 'qualified_company_count')::NUMERIC;
    IF v_qualified IS DISTINCT FROM
         (v_eligibility ->> 'qualified_company_count')::NUMERIC
       OR (v_cost ->> 'execution_cap_microusd')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'execution_cap_microusd')::BIGINT
       OR (v_cost ->> 'cost_per_company_cap_microusd')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'cost_per_company_cap_microusd')::BIGINT
       OR (v_cost ->> 'eligibility_cap_microusd')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'eligibility_cap_microusd')::BIGINT
       OR (v_cost #>> '{execution,settled_microusd}')::BIGINT
          IS DISTINCT FROM (v_eligibility ->> 'settled_microusd')::BIGINT
       OR (v_cost #>> '{execution,reserved_or_uncertain_microusd}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'reserved_or_uncertain_microusd')::BIGINT
       OR (v_cost #>> '{execution,conservative_microusd}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'conservative_microusd')::BIGINT
       OR (v_cost #>> '{execution,inflight_calls}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'execution_inflight_calls')::BIGINT
       OR (v_cost #>> '{execution,uncertain_calls}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'execution_uncertain_calls')::BIGINT
       OR (v_cost #>> '{judge,inflight_calls}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'judge_inflight_calls')::BIGINT
       OR (v_cost #>> '{judge,uncertain_calls}')::BIGINT
          IS DISTINCT FROM
          (v_eligibility ->> 'judge_uncertain_calls')::BIGINT THEN
      RAISE EXCEPTION 'lab_arena_publication_cost_report_mismatch'
        USING ERRCODE = '22023';
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
$lab_arena_integrity_publication_guard$;
ALTER FUNCTION public.lab_arena_integrity_publication_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_integrity_publication_guard_v1()
  FROM PUBLIC;

DROP TRIGGER IF EXISTS lab_arena_integrity_publication_guard
  ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_integrity_publication_guard
  BEFORE UPDATE ON public.lab_arena_rounds
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_integrity_publication_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_integrity_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $lab_arena_integrity_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.integrity_schema.v1',
    'version', 213
  );
$lab_arena_integrity_schema$;
ALTER FUNCTION public.lab_arena_integrity_schema_v1()
  OWNER TO lab_arena_owner;

DO $lab_arena_213_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_prepare_confirmation_bank(TEXT, TEXT, TEXT)',
    'public.lab_arena_open_confirmation(TEXT, JSONB)',
    'public.lab_arena_integrity_schema_v1()'
  ] LOOP
    EXECUTE pg_catalog.format(
      'REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature
    );
    FOREACH role_name IN ARRAY ARRAY[
      'anon', 'authenticated', 'service_role'
    ] LOOP
      IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name
      ) THEN
        EXECUTE pg_catalog.format(
          'REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name
        );
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format(
      'GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature
    );
  END LOOP;
END;
$lab_arena_213_acl$;

COMMENT ON COLUMN public.lab_arena_rounds.confirmation_bank_hash IS
  'Write-once hash of the private five-ICP confirmation bank committed before the public benchmark.';
COMMENT ON COLUMN public.lab_arena_rounds.confirmation_cohort IS
  'Write-once database-validated baseline and top-three main-stage confirmation cohort.';
COMMENT ON COLUMN public.lab_arena_runs.qualification_doc IS
  'Write-once company qualification identities bound to the durable per-ICP score.';
COMMENT ON FUNCTION public.lab_arena_integrity_schema_v1() IS
  'Private service-role feature probe confirming Arena integrity migrations 211 through 213.';

REVOKE ALL ON FUNCTION public.lab_arena__integrity_eligibility(
  TEXT, TEXT, INTEGER[]
) FROM PUBLIC;
NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;
