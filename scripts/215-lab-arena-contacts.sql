-- Opt-in contact qualification for new Arena rounds. No historical data rewrite.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
DO $requires_integrity$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_integrity_schema_v1()') IS NULL THEN
    RAISE EXCEPTION 'apply 213-lab-arena-score-integrity.sql first';
  END IF;
END;
$requires_integrity$;
GRANT CREATE ON SCHEMA public TO lab_arena_owner;


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
           FROM pg_catalog.jsonb_object_keys(v_company)) NOT IN (4, 5)
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
    IF v_company ? 'contact_qualified' THEN
      IF pg_catalog.jsonb_typeof(v_company -> 'contact_qualified')
           IS DISTINCT FROM 'boolean'
         OR ((v_company ->> 'company_qualified')::BOOLEAN
             IS DISTINCT FROM (v_company ->> 'contact_qualified')::BOOLEAN) THEN
        RETURN FALSE;
      END IF;
    ELSIF (SELECT pg_catalog.count(*)
           FROM pg_catalog.jsonb_object_keys(v_company)) <> 4 THEN
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
  IF NEW.configuration_doc ? 'contact_policy' AND (
       NEW.configuration_doc ->> 'contact_policy' IS DISTINCT FROM 'contacts_v1'
       OR NOT v_integrity) THEN
    RAISE EXCEPTION 'lab_arena_contact_policy_invalid' USING ERRCODE = '23514';
  END IF;
  IF NOT v_integrity AND NEW.configuration_doc #>>
       '{scorer_policy,scoring_adapter_version}' = 'qualification_contacts_v3' THEN
    RAISE EXCEPTION 'lab_arena_contact_policy_required' USING ERRCODE = '23514';
  END IF;
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
         IS DISTINCT FROM (CASE WHEN NEW.configuration_doc ->> 'contact_policy' = 'contacts_v1'
           THEN 'qualification_contacts_v3' ELSE 'qualification_integrity_v2' END)
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
  IF NEW.qualification_doc IS NOT NULL THEN
    IF EXISTS (
      SELECT 1 FROM pg_catalog.jsonb_array_elements(NEW.qualification_doc -> 'companies') AS item
      WHERE (item ? 'contact_qualified') IS DISTINCT FROM
        COALESCE(v_round.configuration_doc ->> 'contact_policy' = 'contacts_v1', FALSE)
    ) THEN
      RAISE EXCEPTION 'lab_arena_contact_receipt_policy_mismatch' USING ERRCODE = '23514';
    END IF;
    IF v_round.configuration_doc ->> 'contact_policy' = 'contacts_v1'
       AND NEW.per_icp_score > 0 AND NOT EXISTS (
         SELECT 1 FROM pg_catalog.jsonb_array_elements(NEW.qualification_doc -> 'companies') AS item
         WHERE item ->> 'company_qualified' = 'true'
           AND item ->> 'contact_qualified' = 'true'
           AND item ->> 'duplicate_company' = 'false'
       ) THEN
      RAISE EXCEPTION 'lab_arena_contact_credit_invalid' USING ERRCODE = '23514';
    END IF;
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


CREATE OR REPLACE FUNCTION public.lab_arena_contact_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog
AS $contact_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.contact_schema.v1', 'version', 215
  );
$contact_schema$;
ALTER FUNCTION public.lab_arena_contact_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_contact_schema_v1() FROM PUBLIC;
DO $contact_acl$
DECLARE role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION public.lab_arena_contact_schema_v1() FROM %I', role_name);
    END IF;
  END LOOP;
END;
$contact_acl$;
GRANT EXECUTE ON FUNCTION public.lab_arena_contact_schema_v1() TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;
