-- Exact Nov4 operator reuse of already accepted, identical-source donor evidence.
-- No execution, worker completion signature, provider call, ledger copy or reward activation.
-- The reviewed private payload supplies only existing baseline projection updates.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE arena630_schema_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $patch630_0$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena_runs_terminal_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena_runs_terminal_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='5f795a01c8ed4bf2280597d1f42404c4df46291886be9e7b1a83d35b0e09ea49' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'3d4003101e7638b211885ccdb7fe697106dc85326959ed577276894f99e3dbcf' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena_runs_terminal_v1'; END IF;
 EXECUTE $definition630_0$CREATE OR REPLACE FUNCTION public.lab_arena_runs_terminal_v1()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'pg_catalog'
AS $function$
BEGIN
  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-04'
    AND OLD.submission_id='baseline-2026-11-04' AND OLD.stage=1 AND EXISTS(
     SELECT 1 FROM public.lab_arena_scoring_supersessions h
     WHERE h.round_id=OLD.round_id AND h.request_id='nov04-donor-evidence630-v1'
      AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
      AND h.recovery_doc->>'operation'='baseline_donor_evidence630_v1'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'old'=to_jsonb(OLD)-'updated_at'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'new'=to_jsonb(NEW)-'updated_at') THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND OLD.round_id='arena-2026-11-04'
   AND OLD.submission_id='baseline-2026-11-04' AND OLD.stage=1
   AND OLD.kind IN('execute','score') AND OLD.status='accepted' AND OLD.terminal_cause='accepted'
   AND OLD.icp_position BETWEEN 0 AND 9 AND OLD.scoring_generation_floor=0
   AND NEW.status='failed' AND NEW.terminal_cause='stage_closed'
   AND (to_jsonb(NEW)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
     (to_jsonb(OLD)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
   AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h,
     jsonb_array_elements(h.recovery_doc->'work_items') item
    WHERE h.round_id=OLD.round_id AND h.request_id='nov04-baseline-source621-v1'
     AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
     AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
     AND h.recovery_doc->'superseded_runs'->OLD.run_id=to_jsonb(OLD)
     AND OLD.icp_position=(item->>'icp_position')::INTEGER
     AND OLD.run_id IN(item->>'old_run_id',item->>'old_judge_run_id')
     AND NEW.terminal_doc=jsonb_build_object('operation','baseline_source_supersession_v1',
      'request_id',h.request_id,'new_execution_run_id',item->>'new_run_id')) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;

  IF TG_OP='UPDATE' AND OLD.round_id='arena-2026-11-03'
    AND OLD.kind='score' AND OLD.status='accepted' AND OLD.terminal_cause='accepted'
    AND OLD.submission_id='baseline-2026-11-03' AND OLD.stage=1
    AND OLD.icp_position IN (0,1,2,3,4,5,6,7,9)
    AND OLD.scoring_generation_floor=0
    AND NEW.status='failed' AND NEW.terminal_cause='stage_closed'
    AND (to_jsonb(NEW)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])
    AND NEW.terminal_doc=jsonb_build_object('operation','nov03_baseline_rejudge618_v1',
      'reason','superseded_old_verifier_judgment','request_id','nov03-baseline-rejudge618-v1')
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.request_id='nov03-baseline-rejudge618-v1'
        AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
        AND h.recovery_doc->>'operation'='nov03_baseline_rejudge618_v1'
        AND h.recovery_doc->'superseded_judgments'->OLD.run_id=to_jsonb(OLD)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_scoring_supersessions original_history ON original_history.round_id=h.round_id
       AND original_history.generation_floor=(h.recovery_doc->>'restored_from_generation_floor')::BIGINT
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
       AND h.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)
       AND original_history.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',NEW.per_icp_score,'qualification_doc',NEW.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND NEW.per_icp_score IS NULL AND NEW.qualification_doc IS NULL
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current()
       AND h.backend_pid=pg_backend_pid()
       AND h.old_score_projections->OLD.run_id=jsonb_build_object(
         'per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'lab_arena_runs rows are never deleted' USING ERRCODE = '42501';
  END IF;
  -- Terminal attempts are immutable.
  IF OLD.status IN ('accepted', 'failed') AND (
       NEW.status <> OLD.status OR NEW.terminal_cause IS DISTINCT FROM OLD.terminal_cause
       OR NEW.result_doc IS DISTINCT FROM OLD.result_doc
       OR NEW.output_ref IS DISTINCT FROM OLD.output_ref
       OR NEW.lease_generation <> OLD.lease_generation) THEN
    RAISE EXCEPTION 'terminal attempt is immutable' USING ERRCODE = '42501';
  END IF;
  IF OLD.per_icp_score IS NOT NULL AND NEW.per_icp_score IS DISTINCT FROM OLD.per_icp_score THEN
    RAISE EXCEPTION 'attempt score is write-once' USING ERRCODE = '42501';
  END IF;
  IF OLD.claim_request_id IS NOT NULL AND NEW.claim_request_id IS DISTINCT FROM OLD.claim_request_id THEN
    RAISE EXCEPTION 'claim request binding is write-once' USING ERRCODE = '42501';
  END IF;
  NEW.updated_at := pg_catalog.clock_timestamp();
  RETURN NEW;
END;
$function$
$definition630_0$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'5f795a01c8ed4bf2280597d1f42404c4df46291886be9e7b1a83d35b0e09ea49' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena_runs_terminal_v1'; END IF;
END; $patch630_0$;

DO $patch630_1$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena_integrity_run_guard_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena_integrity_run_guard_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='38a1cbde5b4abcffdc00977f5f30ebc5acdf7ac72e40d51986b4dd9f12a5b45b' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'9c001214f9b067d2b0cb285915a52ff326909f250b55f0ac478fea209d2e70b3' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena_integrity_run_guard_v1'; END IF;
 EXECUTE $definition630_1$CREATE OR REPLACE FUNCTION public.lab_arena_integrity_run_guard_v1()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_round public.lab_arena_rounds;
BEGIN
  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-04'
    AND OLD.submission_id='baseline-2026-11-04' AND OLD.stage=1 AND EXISTS(
     SELECT 1 FROM public.lab_arena_scoring_supersessions h
     WHERE h.round_id=OLD.round_id AND h.request_id='nov04-donor-evidence630-v1'
      AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
      AND h.recovery_doc->>'operation'='baseline_donor_evidence630_v1'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'old'=to_jsonb(OLD)-'updated_at'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'new'=to_jsonb(NEW)-'updated_at') THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_scoring_supersessions original_history ON original_history.round_id=h.round_id
       AND original_history.generation_floor=(h.recovery_doc->>'restored_from_generation_floor')::BIGINT
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
       AND h.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)
       AND original_history.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',NEW.per_icp_score,'qualification_doc',NEW.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND NEW.scoring_generation_floor IS DISTINCT FROM OLD.scoring_generation_floor THEN
    RAISE EXCEPTION 'lab_arena_score_revision_immutable' USING ERRCODE='42501';
  END IF;
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND NEW.per_icp_score IS NULL AND NEW.qualification_doc IS NULL
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current()
       AND h.backend_pid=pg_backend_pid()
       AND h.old_score_projections->OLD.run_id=jsonb_build_object(
         'per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = NEW.round_id;
  IF NOT FOUND THEN RETURN NEW; END IF;
  IF TG_OP='INSERT' THEN
    IF NEW.scoring_generation_floor IS NOT NULL AND NEW.scoring_generation_floor IS DISTINCT FROM
      (CASE WHEN NEW.kind='score' THEN v_round.scoring_generation_floor ELSE 0 END) THEN
      RAISE EXCEPTION 'lab_arena_score_revision_invalid' USING ERRCODE='42501';
    END IF;
    NEW.scoring_generation_floor:=CASE WHEN NEW.kind='score' THEN v_round.scoring_generation_floor ELSE 0 END;
  END IF;
  IF (
       v_round.configuration_doc ->> 'execution_sequence_policy'
         = 'baseline_scored_first_v1'
       AND (
         NEW.stage NOT IN (1, 2)
         OR NEW.icp_position NOT BETWEEN 0 AND
           (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
           + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1
       )
     ) OR (
       v_round.configuration_doc ->> 'execution_sequence_policy'
         IS DISTINCT FROM 'baseline_scored_first_v1'
       AND (
         (NEW.stage = 1 AND NEW.icp_position NOT BETWEEN 0 AND
               (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER - 1)
         OR (NEW.stage = 2 AND NEW.icp_position NOT BETWEEN
               (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER AND
               (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
               + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER - 1)
       )
     ) THEN
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
$function$
$definition630_1$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'38a1cbde5b4abcffdc00977f5f30ebc5acdf7ac72e40d51986b4dd9f12a5b45b' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena_integrity_run_guard_v1'; END IF;
END; $patch630_1$;

DO $patch630_2$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena_operator_hold_score_guard_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena_operator_hold_score_guard_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='df5338f07ad8317c6bb7977377bc166fbf3b8336dfb398f4847e2fe79dee7f61' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'c12bef0857ba2c00c417f21baa589bac63024f51514dcb2a698ec7953f4793d9' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena_operator_hold_score_guard_v1'; END IF;
 EXECUTE $definition630_2$CREATE OR REPLACE FUNCTION public.lab_arena_operator_hold_score_guard_v1()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE
  v_paused BOOLEAN;
BEGIN
  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-04'
    AND OLD.submission_id='baseline-2026-11-04' AND OLD.stage=1 AND EXISTS(
     SELECT 1 FROM public.lab_arena_scoring_supersessions h
     WHERE h.round_id=OLD.round_id AND h.request_id='nov04-donor-evidence630-v1'
      AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
      AND h.recovery_doc->>'operation'='baseline_donor_evidence630_v1'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'old'=to_jsonb(OLD)-'updated_at'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'new'=to_jsonb(NEW)-'updated_at') THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_scoring_supersessions original_history ON original_history.round_id=h.round_id
       AND original_history.generation_floor=(h.recovery_doc->>'restored_from_generation_floor')::BIGINT
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
       AND h.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)
       AND original_history.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',NEW.per_icp_score,'qualification_doc',NEW.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF NEW.per_icp_score IS NULL
     OR NEW.per_icp_score IS NOT DISTINCT FROM OLD.per_icp_score THEN
    RETURN NEW;
  END IF;
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0));
  SELECT operator_paused OR guard_commitment <> '' INTO v_paused
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF COALESCE(v_paused, TRUE) THEN
    RAISE EXCEPTION 'lab_arena_round_progression_paused' USING ERRCODE='55000';
  END IF;
  RETURN NEW;
END;
$function$
$definition630_2$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'df5338f07ad8317c6bb7977377bc166fbf3b8336dfb398f4847e2fe79dee7f61' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena_operator_hold_score_guard_v1'; END IF;
END; $patch630_2$;

DO $patch630_3$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena_runs_participation_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena_runs_participation_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='344f3a1d4cd08914386d6d988d26892640a667297b6cf2f6f7671cfd29da6479' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'8f7ecb2c1e4743fc6e571fa4adb3361a83efc2485af5d66c175a1ae83deb479e' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena_runs_participation_v1'; END IF;
 EXECUTE $definition630_3$CREATE OR REPLACE FUNCTION public.lab_arena_runs_participation_v1()
 RETURNS trigger
 LANGUAGE plpgsql
 SET search_path TO 'pg_catalog'
AS $function$
DECLARE
  v_company_cache JSONB;
  v_original_work BOOLEAN := TRUE;
BEGIN
  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-04'
    AND OLD.submission_id='baseline-2026-11-04' AND OLD.stage=1 AND EXISTS(
     SELECT 1 FROM public.lab_arena_scoring_supersessions h
     WHERE h.round_id=OLD.round_id AND h.request_id='nov04-donor-evidence630-v1'
      AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
      AND h.recovery_doc->>'operation'='baseline_donor_evidence630_v1'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'old'=to_jsonb(OLD)-'updated_at'
      AND h.recovery_doc->'changed_runs'->OLD.run_id->'new'=to_jsonb(NEW)-'updated_at') THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP = 'INSERT' THEN
    IF NEW.participation_accepted_at IS NOT NULL THEN
      RAISE EXCEPTION 'participation timestamp is database-owned'
        USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
  END IF;
  IF NEW.participation_accepted_at IS DISTINCT FROM
      OLD.participation_accepted_at THEN
    RAISE EXCEPTION 'participation timestamp is immutable'
      USING ERRCODE = '42501';
  END IF;
  IF OLD.participation_accepted_at IS NOT NULL AND (
      NEW.runner_hotkey IS DISTINCT FROM OLD.runner_hotkey
      OR NEW.round_id IS DISTINCT FROM OLD.round_id) THEN
    RAISE EXCEPTION 'participation identity is immutable'
      USING ERRCODE = '42501';
  END IF;

  -- A score with company refs uses the database-generated claim cache as its
  -- proof of original work. At least one miss means this lease produced a new
  -- judgment. An all-hit or malformed cache is reuse and earns no timestamp.
  -- Scores from before the company-judgment protocol have NULL refs and keep
  -- the original participation behavior. Execute runs are also unchanged.
  IF OLD.kind = 'score' AND OLD.company_judgment_refs IS NOT NULL THEN
    v_original_work := FALSE;
    v_company_cache := OLD.claim_response -> 'company_judgment_cache';
    IF pg_catalog.jsonb_typeof(OLD.company_judgment_refs) = 'array'
       AND pg_catalog.jsonb_typeof(v_company_cache) = 'object'
       AND v_company_cache ->> 'schema_version' =
           'leadpoet.lab_arena.company_judgment_lease.v1'
       AND pg_catalog.jsonb_typeof(v_company_cache -> 'hits') = 'array'
       AND pg_catalog.jsonb_typeof(v_company_cache -> 'misses') = 'array' THEN
      v_original_work :=
        pg_catalog.jsonb_array_length(v_company_cache -> 'misses') > 0
        AND (NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds frozen
          WHERE frozen.round_id=OLD.round_id AND frozen.configuration_doc->>'company_judgment_policy'='company_judgment_v1')
          OR COALESCE(NEW.result_doc->>'company_judgments_stored','0') ~ '^[1-5]$');
    END IF;
  END IF;

  -- The completion RPC remains authoritative for the accepted transition,
  -- lease owner, generation, output, and accounting checks. This trigger only
  -- decides whether that accepted transition contains original work.
  IF v_original_work
      AND OLD.status = 'leased' AND NEW.status = 'accepted'
      AND NEW.terminal_cause = 'accepted'
      AND OLD.runner_hotkey IS NOT NULL
      AND NEW.runner_hotkey IS NOT DISTINCT FROM OLD.runner_hotkey
      AND NEW.round_id IS NOT DISTINCT FROM OLD.round_id
      AND OLD.lease_token_hash IS NOT NULL
      AND OLD.claim_request_id IS NOT NULL
      AND (NEW.result_doc ->> 'schema_version') IS DISTINCT FROM
          'leadpoet.lab_arena.cached_run_result.v1' THEN
    NEW.participation_accepted_at := pg_catalog.clock_timestamp();
  END IF;
  RETURN NEW;
END;
$function$
$definition630_3$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'344f3a1d4cd08914386d6d988d26892640a667297b6cf2f6f7671cfd29da6479' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena_runs_participation_v1'; END IF;
END; $patch630_3$;

DO $patch630_4$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena__source_correction621_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena__source_correction621_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='5b24291b3e5c87a94741a5b168a76a115edbe1eb0f150999effdd99fffa507e2' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'d599eee2885087984be0abe6959bb61767f8b9759aceb8ecfe87ad0f3325f367' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena__source_correction621_v1'; END IF;
 EXECUTE $definition630_4$CREATE OR REPLACE FUNCTION public.lab_arena__source_correction621_v1(p_run lab_arena_runs)
 RETURNS boolean
 LANGUAGE plpgsql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
DECLARE r public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 item JSONB; original public.lab_arena_runs;
BEGIN
 IF p_run.round_id IS DISTINCT FROM 'arena-2026-11-04'
  OR p_run.submission_id IS DISTINCT FROM 'baseline-2026-11-04'
  OR p_run.kind IS DISTINCT FROM 'execute' OR p_run.stage IS DISTINCT FROM 1
  OR p_run.icp_position NOT BETWEEN 0 AND 9 OR p_run.scoring_generation_floor<>0
  OR p_run.assignment_id IS DISTINCT FROM p_run.round_id||':'||p_run.submission_id||':1:'||p_run.icp_position
  OR p_run.run_id IS DISTINCT FROM p_run.assignment_id||':'||p_run.attempt
  OR p_run.champion_restart_required THEN RETURN FALSE; END IF;
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_run.round_id;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions source621
 WHERE round_id=p_run.round_id AND request_id='nov04-baseline-source621-v1'
  AND recovery_doc->>'operation'='baseline_source_supersession_v1'
  AND (source621.new_round_doc->'configuration_doc'=r.configuration_doc OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions correction630 WHERE correction630.round_id=r.round_id AND correction630.request_id='nov04-donor-evidence630-v1' AND correction630.recovery_doc->>'operation'='baseline_donor_evidence630_v1' AND correction630.old_round_doc->'configuration_doc'=source621.new_round_doc->'configuration_doc' AND correction630.new_round_doc->'configuration_doc'=r.configuration_doc))
  AND new_round_doc->'participants'=r.participants
  AND new_round_doc->>'champion_hotkey'=r.champion_hotkey
  AND new_round_doc->>'champion_submission_id'=r.champion_submission_id;
 IF NOT FOUND OR r.scoring_generation_floor<>0
  OR p_run.miner_hotkey IS DISTINCT FROM r.configuration_doc->>'baseline_hotkey'
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_submissions s
   WHERE s.submission_id=p_run.submission_id AND s.status='frozen' AND s.is_king
    AND s.miner_hotkey=p_run.miner_hotkey
    AND (to_jsonb(s)-'updated_at')=h.recovery_doc->'new_submission_doc') THEN RETURN FALSE; END IF;
 SELECT value INTO item FROM jsonb_array_elements(h.recovery_doc->'work_items')
 WHERE value->>'icp_position'=p_run.icp_position::TEXT
  AND p_run.attempt BETWEEN (value->>'new_attempt')::INTEGER
   AND (value->>'new_attempt')::INTEGER+(r.configuration_doc->>'max_attempts_per_assignment')::INTEGER-1;
 IF NOT FOUND THEN RETURN FALSE; END IF;
 -- The frozen normal attempt budget starts at this source generation's root.
 -- Only complete failed infrastructure lineage can consume its next attempt.
 IF p_run.attempt>(item->>'new_attempt')::INTEGER AND NOT (
   SELECT count(*)=p_run.attempt-(item->>'new_attempt')::INTEGER
    AND coalesce(bool_and(parent.status='failed' AND parent.kind='execute'
     AND parent.terminal_cause IN('provider_error','lease_expired')
     AND parent.output_ref IS NULL AND NOT parent.champion_restart_required
     AND (parent.terminal_cause='lease_expired'
       OR parent.result_doc->>'terminal_status'=parent.terminal_cause)
     AND parent.run_id=parent.assignment_id||':'||parent.attempt),FALSE)
   FROM public.lab_arena_runs parent WHERE parent.assignment_id=p_run.assignment_id
    AND parent.attempt BETWEEN (item->>'new_attempt')::INTEGER AND p_run.attempt-1
 ) THEN RETURN FALSE; END IF;
 SELECT * INTO original FROM public.lab_arena_runs WHERE run_id=item->>'old_run_id';
 RETURN COALESCE(original.status='failed' AND original.terminal_cause='stage_closed'
  AND original.terminal_doc=jsonb_build_object('operation','baseline_source_supersession_v1',
    'request_id',h.request_id,'new_execution_run_id',item->>'new_run_id')
  AND (to_jsonb(original)-ARRAY['status','terminal_cause','terminal_doc','updated_at'])=
    ((h.recovery_doc->'superseded_runs'->original.run_id)-
      ARRAY['status','terminal_cause','terminal_doc','updated_at']),FALSE);
END;
$function$
$definition630_4$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'5b24291b3e5c87a94741a5b168a76a115edbe1eb0f150999effdd99fffa507e2' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena__source_correction621_v1'; END IF;
END; $patch630_4$;

DO $patch630_5$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena__retired_source621_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena__retired_source621_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='36507e89c967e86880e48e8cac3a91c3b9374e1d435769ad9e213cb05e80fe53' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'f4dcd76d47341ecb8e643a1a3a756848b3477d052966bfc7d49c68a9c87d24d5' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena__retired_source621_v1'; END IF;
 EXECUTE $definition630_5$CREATE OR REPLACE FUNCTION public.lab_arena__retired_source621_v1(p_run lab_arena_runs)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
 SELECT COALESCE(p_run.round_id='arena-2026-11-04' AND p_run.submission_id='baseline-2026-11-04'
  AND p_run.stage=1 AND p_run.kind IN('execute','score') AND p_run.scoring_generation_floor=0
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
   WHERE h.round_id=p_run.round_id AND h.request_id='nov04-baseline-source621-v1'
    AND h.recovery_doc->>'operation'='baseline_source_supersession_v1'
    AND h.recovery_doc->'retired_source_run_hashes' ? p_run.run_id
    AND (h.new_round_doc->'configuration_doc'=r.configuration_doc OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions correction630 WHERE correction630.round_id=r.round_id AND correction630.request_id='nov04-donor-evidence630-v1' AND correction630.recovery_doc->>'operation'='baseline_donor_evidence630_v1' AND correction630.old_round_doc->'configuration_doc'=h.new_round_doc->'configuration_doc' AND correction630.new_round_doc->'configuration_doc'=r.configuration_doc))
    AND h.old_round_doc#>>'{configuration_doc,reasoning_baseline_source_sha256}' IS DISTINCT FROM
      h.new_round_doc#>>'{configuration_doc,reasoning_baseline_source_sha256}'),FALSE);
$function$
$definition630_5$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'36507e89c967e86880e48e8cac3a91c3b9374e1d435769ad9e213cb05e80fe53' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena__retired_source621_v1'; END IF;
END; $patch630_5$;

DO $patch630_6$ DECLARE t OID; d TEXT; m JSONB;
BEGIN
 SELECT p.oid,pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT t,d,m FROM pg_proc p
 WHERE p.pronamespace='public'::regnamespace AND p.proname='lab_arena_scoring_policy_history_v1';
 IF pg_get_userbyid((m->>'proowner')::OID)<>'lab_arena_owner'
  OR (SELECT p.proacl::TEXT FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' THEN RAISE EXCEPTION 'Donor630 function ACL differs: lab_arena_scoring_policy_history_v1'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='1bc738f6e73c29ac756106ff233af894915a492547334f6ecb66815654f6031a' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'873cdf9c0e5ad4a6427617fa7377a64af17a83d631e24e7ca9ac81c476eeadf3' THEN RAISE EXCEPTION 'Donor630 function source differs: lab_arena_scoring_policy_history_v1'; END IF;
 EXECUTE $definition630_6$CREATE OR REPLACE FUNCTION public.lab_arena_scoring_policy_history_v1(p_round_id text)
 RETURNS jsonb
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
 SELECT jsonb_build_object('round_id',p_round_id,'source_corrections',coalesce((
  SELECT jsonb_agg(jsonb_build_object(
   'operation','baseline_source_supersession_v1','request_id',source_history.request_id,
   'request_hash',source_history.request_hash,'round_id',source_history.round_id,
   'submission_id',baseline.submission_id,'miner_hotkey',baseline.miner_hotkey,
   'source_ref',baseline.source_ref,'source_size_bytes',baseline.source_size_bytes,
   'source_archive_sha256',baseline.source_archive_sha256,'source_normalized_sha256',baseline.source_normalized_sha256,
   'baseline_source_url',r.configuration_doc->>'baseline_source_url',
   'reasoning_baseline_source_sha256',r.configuration_doc->>'reasoning_baseline_source_sha256',
   'run_ids',coalesce((SELECT jsonb_agg(a.run_id ORDER BY a.run_id)
     FROM jsonb_array_elements(source_history.recovery_doc->'work_items') item
     JOIN public.lab_arena_runs a ON a.round_id=source_history.round_id
       AND a.submission_id=baseline.submission_id AND a.icp_position=(item->>'icp_position')::INTEGER
     WHERE a.kind='execute' AND public.lab_arena__source_correction621_v1(a)),'[]'::JSONB))
    ORDER BY source_history.generation_floor)
  FROM public.lab_arena_scoring_supersessions source_history
  JOIN public.lab_arena_rounds r ON r.round_id=source_history.round_id
  JOIN public.lab_arena_submissions baseline ON baseline.round_id=r.round_id
    AND baseline.submission_id='baseline-2026-11-04' AND baseline.is_king AND baseline.status='frozen'
  WHERE source_history.round_id=p_round_id AND source_history.round_id='arena-2026-11-04'
   AND source_history.request_id='nov04-baseline-source621-v1'
   AND source_history.recovery_doc->>'operation'='baseline_source_supersession_v1'
   AND (source_history.new_round_doc->'configuration_doc'=r.configuration_doc OR EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions correction630 WHERE correction630.round_id=r.round_id AND correction630.request_id='nov04-donor-evidence630-v1' AND correction630.recovery_doc->>'operation'='baseline_donor_evidence630_v1' AND correction630.old_round_doc->'configuration_doc'=source_history.new_round_doc->'configuration_doc' AND correction630.new_round_doc->'configuration_doc'=r.configuration_doc))
   AND source_history.new_round_doc->'participants'=r.participants
   AND source_history.new_round_doc->>'champion_hotkey'=r.champion_hotkey
   AND source_history.new_round_doc->>'champion_submission_id'=r.champion_submission_id
   AND source_history.recovery_doc->'new_submission_doc'=to_jsonb(baseline)-'updated_at'
 ),'[]'::JSONB),'revisions',coalesce(jsonb_agg(jsonb_build_object(
  'previous_generation_floor',coalesce((h.old_round_doc->>'scoring_generation_floor')::BIGINT,0),
  'generation_floor',CASE WHEN h.recovery_doc->>'operation' IN('nov03_forward_only_verifier_v1','nov03_admission_review605_v1','nov03_admit_reviewed_originals606_v1','nov03_baseline8_corrected_judge607_v1','nov03_three_unfinished_executions609_v1','nov03_baseline8_checkpoint_continuation610_v1','nov03_g3_natural_expiry_executions611_v1','nov03_mychn_checkpoint_continuation612_v1','nov03_filson_checkpoint_continuation613_v1','nov03_captify_checkpoint_continuation614_v1','baseline_source_supersession_v1','baseline_source_replacement_plan_v1','baseline_donor_evidence630_v1') THEN (h.new_round_doc->>'scoring_generation_floor')::BIGINT ELSE h.generation_floor END,
  'receipt_generation',h.generation_floor,'old_scorer_policy',h.old_round_doc#>'{configuration_doc,scorer_policy}',
  'new_scorer_policy',h.new_round_doc#>'{configuration_doc,scorer_policy}',
  'recovery',CASE WHEN h.recovery_doc IS NOT NULL AND h.recovery_doc->>'operation'<>'restore_recorded_result_v1' THEN
   jsonb_build_object('operation',h.recovery_doc->>'operation','excluded_participants',coalesce((
    SELECT jsonb_agg(jsonb_build_object('submission_id',p->>'submission_id','miner_hotkey',p->>'miner_hotkey',
      'status','provider_blocked','evaluation','not_evaluated','reason','openrouter_zdr_route_unavailable'))
    FROM jsonb_array_elements(h.recovery_doc->'excluded_participants')p),'[]')) END,
  'restoration',CASE WHEN h.recovery_doc->>'operation'='restore_recorded_result_v1' THEN
   jsonb_build_object('selected_generation_floor',h.recovery_doc->'selected_generation_floor',
    'excluded_participants',coalesce((SELECT jsonb_agg(jsonb_build_object('submission_id',p->>'submission_id',
     'miner_hotkey',p->>'miner_hotkey','status',p->>'status','evaluation','not_evaluated','reason',p->>'reason'))
     FROM jsonb_array_elements(h.recovery_doc->'excluded_participants')p),'[]')) END
 ) ORDER BY h.generation_floor),'[]')) FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=p_round_id;
$function$
$definition630_6$;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=t) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(t),'sha256'),'hex')<>'1bc738f6e73c29ac756106ff233af894915a492547334f6ecb66815654f6031a' THEN RAISE EXCEPTION 'Donor630 function readback differs: lab_arena_scoring_policy_history_v1'; END IF;
END; $patch630_6$;

DO $data630$
DECLARE p JSONB; r public.lab_arena_rounds; n public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 h public.lab_arena_scoring_supersessions; item JSONB; change RECORD; oldrun public.lab_arena_runs;
 newrun public.lab_arena_runs; donor public.lab_arena_runs; judge public.lab_arena_runs;
 hashes JSONB; revision BIGINT; request_hash TEXT; positions INTEGER[]; expected_count INTEGER;
BEGIN
 IF octet_length(coalesce(current_setting('lab_arena.donor630_payload',TRUE),'')) NOT BETWEEN 2 AND 16777216 THEN
  RAISE EXCEPTION 'Donor630 reviewed private payload missing'; END IF;
 p:=current_setting('lab_arena.donor630_payload')::JSONB;
 IF p->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.baseline_donor630.v1'
  OR p->>'source_normalized_sha256' IS DISTINCT FROM 'sha256:1047b383f4b32dbc38d19e5f052384174d4a9a89a7c48b4620cecb18714ed06d' THEN
  RAISE EXCEPTION 'Donor630 payload identity differs'; END IF;
 SELECT array_agg(value::INTEGER ORDER BY value::INTEGER) INTO positions FROM jsonb_array_elements_text(p->'positions');
 -- The coordinator must pin the user's selected alternative in the payload hash.
 IF positions IS DISTINCT FROM ARRAY[0,1,2,3,4,5,6,7,8,9] THEN
  RAISE EXCEPTION 'Donor630 position selection differs'; END IF;
 expected_count:=cardinality(positions);
 request_hash:='sha256:'||encode(extensions.digest(p::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-04' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id AND request_id='nov04-donor-evidence630-v1';
 IF FOUND THEN
  IF h.request_hash IS DISTINCT FROM request_hash OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Donor630 replay differs'; END IF;
  RETURN;
 END IF;
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR UPDATE;
 IF r.status<>'scored' OR r.scoring_generation_floor<>0 OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL
  OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL OR r.effective_reward_epoch IS NOT NULL
  OR to_jsonb(r) IS DISTINCT FROM p->'round'
  OR NOT c.operator_paused OR c.guard_generation IS DISTINCT FROM (p#>>'{guard,guard_generation}')::BIGINT
  OR c.actor_ref IS DISTINCT FROM p#>>'{guard,actor_ref}' OR c.pause_reason IS DISTINCT FROM p#>>'{guard,pause_reason}'
  OR c.guard_commitment<>'' OR c.owner_commitment<>'' OR c.guard_expires_at IS NOT NULL
  OR c.candidate_commit<>'' OR c.restart_scope<>'' OR c.restart_phase<>'' OR c.captured_leases<>'[]'::JSONB
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND a.status IN('pending','leased','submitted'))
  THEN RAISE EXCEPTION 'Donor630 round/owned hold/deadline differs'; END IF;
 SELECT jsonb_object_agg(run_id,encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex')) INTO hashes
 FROM public.lab_arena_runs a WHERE round_id=r.round_id;
 IF hashes IS DISTINCT FROM p->'run_hashes' OR (SELECT count(*) FROM jsonb_object_keys(p->'changed_runs'))<>expected_count*2
  OR jsonb_array_length(p->'items')<>expected_count THEN RAISE EXCEPTION 'Donor630 original run manifest differs'; END IF;
 IF (SELECT count(*) FROM public.lab_arena_submissions s WHERE s.round_id=r.round_id
  AND s.submission_id IN('baseline-2026-11-04','sub-0510cb617eabd69f2f0aebeb69b37fe7') AND s.status='frozen'
  AND s.source_normalized_sha256='sha256:1047b383f4b32dbc38d19e5f052384174d4a9a89a7c48b4620cecb18714ed06d')<>2 THEN
  RAISE EXCEPTION 'Donor630 frozen source equivalence differs'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(p->'items') LOOP
  IF NOT (item->>'position')::INTEGER=ANY(positions) THEN RAISE EXCEPTION 'Donor630 extra position'; END IF;
  SELECT * INTO STRICT donor FROM public.lab_arena_runs WHERE run_id=item->>'donor_execution' FOR SHARE;
  SELECT * INTO STRICT judge FROM public.lab_arena_runs WHERE run_id=item->>'donor_judgment' FOR SHARE;
  IF donor.round_id<>r.round_id OR donor.submission_id<>'sub-0510cb617eabd69f2f0aebeb69b37fe7'
   OR donor.kind<>'execute' OR donor.stage<>2 OR donor.icp_position<>(item->>'position')::INTEGER
   OR donor.status<>'accepted' OR donor.terminal_cause<>'accepted' OR donor.per_icp_score IS NULL OR donor.output_ref IS NULL
   OR judge.scored_run_id<>donor.run_id OR judge.kind<>'score' OR judge.status<>'accepted' OR judge.terminal_cause<>'accepted'
   OR judge.scoring_generation_floor<>0 OR judge.submission_id<>donor.submission_id OR judge.stage<>donor.stage
   OR judge.icp_position<>donor.icp_position OR item->'donor_refs' IS DISTINCT FROM judge.company_judgment_refs
   OR NOT public.lab_arena__judge_calls_clear_v1(donor.assignment_id)
   OR NOT public.lab_arena__judge_calls_clear_v1(judge.assignment_id) THEN RAISE EXCEPTION 'Donor630 donor authority differs'; END IF;
  FOR change IN SELECT key,value FROM jsonb_each(p->'changed_runs') x WHERE x.key IN(item->>'baseline_execution',item->>'baseline_judgment') LOOP
   SELECT * INTO STRICT oldrun FROM public.lab_arena_runs WHERE run_id=change.key FOR UPDATE;
   newrun:=jsonb_populate_record(NULL::public.lab_arena_runs,change.value->'new');
   IF to_jsonb(oldrun)-'updated_at' IS DISTINCT FROM change.value->'old'
    OR oldrun.round_id<>r.round_id OR oldrun.submission_id<>'baseline-2026-11-04' OR oldrun.stage<>1
    OR oldrun.icp_position<>(item->>'position')::INTEGER OR oldrun.scoring_generation_floor<>0
    OR NOT public.lab_arena__judge_calls_clear_v1(oldrun.assignment_id)
    OR newrun.terminal_doc->>'operation'<>'baseline_donor_evidence630_v1'
    OR newrun.terminal_doc->>'donor_execution_run_id'<>donor.run_id OR newrun.terminal_doc->>'donor_judgment_run_id'<>judge.run_id
    OR newrun.terminal_doc->'signed_worker_completion'<>'false'::JSONB
    OR newrun.terminal_doc->'projection_not_new_execution'<>'true'::JSONB THEN RAISE EXCEPTION 'Donor630 original/projection identity differs'; END IF;
   IF oldrun.kind='execute' THEN
    IF oldrun.status<>'accepted' OR oldrun.terminal_cause<>'accepted'
     OR ((change.value->'new')-ARRAY['output_ref','per_icp_score','qualification_doc','terminal_doc']) IS DISTINCT FROM
        ((change.value->'old')-ARRAY['output_ref','per_icp_score','qualification_doc','terminal_doc'])
     OR newrun.output_ref<>donor.output_ref OR newrun.per_icp_score IS DISTINCT FROM donor.per_icp_score
     OR newrun.qualification_doc IS DISTINCT FROM donor.qualification_doc THEN RAISE EXCEPTION 'Donor630 execution projection differs'; END IF;
   ELSIF oldrun.kind='score' THEN
    IF newrun.status<>'accepted' OR newrun.terminal_cause<>'accepted' OR newrun.scored_run_id<>item->>'baseline_execution'
     OR ((change.value->'new')-ARRAY['status','terminal_cause','output_ref','result_doc','company_judgment_refs','claim_response','terminal_doc']) IS DISTINCT FROM
        ((change.value->'old')-ARRAY['status','terminal_cause','output_ref','result_doc','company_judgment_refs','claim_response','terminal_doc'])
     OR (newrun.claim_response-'company_judgment_cache') IS DISTINCT FROM (oldrun.claim_response-'company_judgment_cache')
     OR newrun.company_judgment_refs IS DISTINCT FROM judge.company_judgment_refs
     OR newrun.output_ref IS DISTINCT FROM item->>'derived_output_ref'
     OR newrun.output_ref NOT LIKE 'arena/arena-2026-11-04/scores/operator-donor630/%'
     OR newrun.claim_response#>'{company_judgment_cache,misses}'<>'[]'::JSONB
     OR jsonb_array_length(newrun.claim_response#>'{company_judgment_cache,hits}')<>jsonb_array_length(judge.company_judgment_refs)
     OR NOT public.lab_arena__company_judgment_refs_v1(r,newrun.company_judgment_refs)
     OR EXISTS(SELECT 1 FROM jsonb_array_elements(newrun.claim_response#>'{company_judgment_cache,hits}') hit
      WHERE NOT EXISTS(SELECT 1 FROM public.lab_arena_company_judgments cached
       WHERE cached.cache_key=hit->>'cache_key' AND cached.authority_slot=(hit->>'authority_slot')::BIGINT
        AND cached.company_input_hash=hit->>'company_input_hash' AND cached.evidence_hash=hit->>'evidence_hash'
        AND cached.evidence_doc=hit->'evidence_doc')) THEN RAISE EXCEPTION 'Donor630 cache-only judgment projection differs'; END IF;
   ELSE RAISE EXCEPTION 'Donor630 unexpected projection kind'; END IF;
   IF (public.lab_arena_icp_cost_eligibility(r.round_id,oldrun.submission_id,oldrun.icp_position,(item->>'qualified')::INTEGER)->>'eligible')::BOOLEAN IS NOT TRUE
    OR (public.lab_arena_icp_cost_eligibility(r.round_id,donor.submission_id,donor.icp_position,(item->>'qualified')::INTEGER)->>'eligible')::BOOLEAN IS NOT TRUE THEN
    RAISE EXCEPTION 'Donor630 equivalent effective cost gate differs'; END IF;
  END LOOP;
 END LOOP;
 n:=r; n.status_generation:=r.status_generation+1;
 n.configuration_doc:=jsonb_set(jsonb_set(r.configuration_doc,'{schedule,final_scoring_close}','"2026-11-05T02:55:00Z"'),'{schedule,publication_deadline}','"2026-11-05T03:00:00Z"');
 SELECT jsonb_set(r.stage1_scoring_plan_doc,'{work_items}',jsonb_agg(CASE WHEN (saved->>'icp_position')::INTEGER=ANY(positions)
  THEN jsonb_set(saved,'{output_ref}',to_jsonb((SELECT e.value->'new'->>'output_ref' FROM jsonb_each(p->'changed_runs')e
   WHERE e.key=saved->>'scored_run_id' AND e.value->'new'->>'kind'='execute'))) ELSE saved END ORDER BY ord))
 INTO n.stage1_scoring_plan_doc FROM jsonb_array_elements(r.stage1_scoring_plan_doc->'work_items') WITH ORDINALITY rows(saved,ord);
 SELECT coalesce(max(generation_floor),0)+1 INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,transaction_id,backend_pid,
 old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,revision,'nov04-donor-evidence630-v1',request_hash,'nov04-user-approved-donor-evidence',txid_current(),pg_backend_pid(),
 to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at','{}',jsonb_build_object('operation','baseline_donor_evidence630_v1','scoring_generation_floor',0,
 'positions',p->'positions','changed_runs',p->'changed_runs','items',p->'items','source_normalized_sha256',p->'source_normalized_sha256','proof_sha256',p->'proof_sha256'));
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,status_generation=n.status_generation,
  stage1_scoring_plan_doc=n.stage1_scoring_plan_doc WHERE round_id=r.round_id;
 FOR change IN SELECT key,value FROM jsonb_each(p->'changed_runs') LOOP
  newrun:=jsonb_populate_record(NULL::public.lab_arena_runs,change.value->'new');
  UPDATE public.lab_arena_runs SET status=newrun.status,terminal_cause=newrun.terminal_cause,output_ref=newrun.output_ref,
   per_icp_score=newrun.per_icp_score,qualification_doc=newrun.qualification_doc,result_doc=newrun.result_doc,
   company_judgment_refs=newrun.company_judgment_refs,claim_response=newrun.claim_response,terminal_doc=newrun.terminal_doc WHERE run_id=change.key;
 END LOOP;
 IF (SELECT count(*) FROM public.lab_arena_runs WHERE round_id=r.round_id)<>(SELECT count(*) FROM jsonb_object_keys(p->'run_hashes'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=r.round_id AND NOT(p->'changed_runs'?a.run_id)
    AND encode(extensions.digest(to_jsonb(a)::TEXT,'sha256'),'hex') IS DISTINCT FROM p->'run_hashes'->>a.run_id) THEN
  RAISE EXCEPTION 'Donor630 changed unselected work'; END IF;
END;
$data630$;
DO $restore630$ BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena630_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena630_schema_acl) THEN RAISE EXCEPTION 'Donor630 schema ACL differs'; END IF;
END; $restore630$;
NOTIFY pgrst,'reload schema';
COMMIT;
