-- Preserve only bounded, source-bound research classification from the existing
-- miner-funded code review. No old result, ledger entry, or round is rewritten.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $research_review556$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '7db9233a48f24be3cf2fa61572546774cccefc1d69d95af2ad3c5dc10bea8083';
  v_postimage CONSTANT TEXT := '9a553aacae9312c94023a6f768b6a6b9e31f2dde07051ce406a135332b95b3c0';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_old TEXT;
  v_new TEXT;
  v_identity JSONB;
  v_metadata JSONB;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig),
    pg_catalog.to_jsonb(p)-'prosrc'
  INTO v_definition,v_identity,v_metadata
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena review556 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review556 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;

  v_old:=$old556a$  v_retryable BOOLEAN;
BEGIN$old556a$;
  v_new:=$new556a$  v_retryable BOOLEAN;
  v_research_doc JSONB := '{}'::JSONB;
  v_research_input JSONB;
  v_research_classification TEXT;
BEGIN$new556a$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review556 declaration anchor differs' USING ERRCODE='55000'; END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old556b$'code_review_round_closed', 'code_review_submission_ineligible',
         'review_response_invalid'$old556b$;
  v_new:=$new556b$'code_review_round_closed', 'code_review_submission_ineligible',
         'review_response_invalid', 'code_review_research_unverified'$new556b$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review556 error anchor differs' USING ERRCODE='55000'; END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old556c$'finding_evidence_mismatch', 'finding_explanation'$old556c$;
  v_new:=$new556c$'finding_evidence_mismatch', 'finding_explanation',
             'research_keys', 'research_binding', 'research_classification',
             'research_evidence', 'research_evidence_mismatch',
             'research_limitations'$new556c$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review556 reason anchor differs' USING ERRCODE='55000'; END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old556d$  v_effective_status := p_status;$old556d$;
  v_new:=$new556d$  -- Only an active future policy may attach research metadata. Provider failures
  -- may have no assessment, but an admitted pass must be source-bound.
  IF v_round.configuration_doc ->> 'reasoning_capture_policy' =
       'full_generated_thinking_v1' THEN
    IF p_status='passed' AND NOT p_review_doc ? 'research_control' THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
  ELSIF p_review_doc ? 'research_control' THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
  END IF;
  IF p_review_doc ? 'research_control' THEN
    v_research_input:=p_review_doc -> 'research_control';
    IF pg_catalog.jsonb_typeof(v_research_input) IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    IF (SELECT count(*) FROM pg_catalog.jsonb_object_keys(v_research_input))<>8
       OR EXISTS (SELECT 1 FROM pg_catalog.jsonb_object_keys(v_research_input) AS field(name)
         WHERE field.name NOT IN (
           'classification','policy_id','policy_hash','reviewer_version',
           'source_archive_sha256','source_manifest_sha256',
           'evidence_claims','evidence_count'))
       OR v_research_input -> 'policy_id' IS DISTINCT FROM
          '"model_directed_research_v1"'::JSONB
       OR v_research_input -> 'reviewer_version' IS DISTINCT FROM
          '"full_source_reviewer_research_v1"'::JSONB
       OR pg_catalog.jsonb_typeof(v_research_input -> 'policy_hash') IS DISTINCT FROM 'string'
       OR COALESCE(v_research_input ->> 'policy_hash','') !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_research_input -> 'source_archive_sha256') IS DISTINCT FROM 'string'
       OR COALESCE(v_research_input ->> 'source_archive_sha256','') !~ '^sha256:[0-9a-f]{64}$'
       OR v_research_input ->> 'source_archive_sha256' IS DISTINCT FROM
          v_submission.source_archive_sha256
       OR pg_catalog.jsonb_typeof(v_research_input -> 'source_manifest_sha256') IS DISTINCT FROM 'string'
       OR COALESCE(v_research_input ->> 'source_manifest_sha256','') !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_research_input -> 'classification') IS DISTINCT FROM 'string'
       OR v_research_input ->> 'classification' NOT IN
          ('model_directed','workflow','unverified')
       OR pg_catalog.jsonb_typeof(v_research_input -> 'evidence_claims') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_research_input -> 'evidence_count') IS DISTINCT FROM 'number'
       OR COALESCE(v_research_input ->> 'evidence_count','') !~ '^([1-9]|1[0-6])$'
    THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    v_research_classification:=v_research_input ->> 'classification';
    IF (v_research_classification='model_directed' AND
          v_research_input -> 'evidence_claims' IS DISTINCT FROM
            '["finish_continue","observation_to_action"]'::JSONB)
       OR (v_research_classification='workflow' AND
          v_research_input -> 'evidence_claims' IS DISTINCT FROM '["fixed_control"]'::JSONB)
       OR (v_research_classification='unverified' AND
          v_research_input -> 'evidence_claims' IS DISTINCT FROM '["uncertainty"]'::JSONB)
       OR (p_status='passed' AND v_research_classification<>'model_directed')
       OR (p_review_doc ->> 'error_code' = 'code_review_research_unverified'
           AND (p_status<>'error' OR v_research_classification<>'unverified'))
    THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    v_research_doc:=pg_catalog.jsonb_build_object('research_control',
      pg_catalog.jsonb_build_object(
        'classification',v_research_classification,
        'policy_id',v_research_input ->> 'policy_id',
        'policy_hash',v_research_input ->> 'policy_hash',
        'reviewer_version',v_research_input ->> 'reviewer_version',
        'source_archive_sha256',v_research_input ->> 'source_archive_sha256',
        'source_manifest_sha256',v_research_input ->> 'source_manifest_sha256',
        'evidence_claims',v_research_input -> 'evidence_claims',
        'evidence_count',v_research_input -> 'evidence_count'));
  ELSIF p_review_doc ->> 'error_code' = 'code_review_research_unverified' THEN
    RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
  END IF;
  v_effective_status := p_status;$new556d$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review556 validation anchor differs' USING ERRCODE='55000'; END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old556e$    ELSE v_diagnostic
  END;$old556e$;
  v_new:=$new556e$    ELSE v_diagnostic
  END || v_research_doc;$new556e$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review556 result anchor differs' USING ERRCODE='55000'; END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  v_old:=$old556f$      || v_diagnostic;$old556f$;
  v_new:=$new556f$      || v_diagnostic || v_research_doc;$new556f$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review556 late-result anchor differs' USING ERRCODE='55000'; END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);

  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review556 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
      <>v_postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
        FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review556 readback differs' USING ERRCODE='55000';
  END IF;
END;
$research_review556$;

COMMIT;
