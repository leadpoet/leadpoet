-- Migration 465: retain terminal company judgments from a failed score attempt.
-- Reuse the existing immutable cache and signed completion. Partial evidence
-- does not accept the ICP, create a score, or change retry/accounting policy.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Keep all current completion policy patches. Permit failed output only for
-- company-quality score attempts; the v3 wrapper binds every evidence row to
-- its live reservation before this existing RPC completes the attempt.
DO $partial_company_output$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$OR (p_terminal_cause <> 'accepted' AND COALESCE(p_output_ref, '') <> '')$old$;
  v_new TEXT := $new$OR (p_terminal_cause <> 'accepted' AND COALESCE(p_output_ref, '') <> ''
         AND NOT (
           p_terminal_cause IN ('judge_error', 'judge_timeout', 'credential_error')
           AND pg_catalog.char_length(p_output_ref) BETWEEN 1 AND 1024
           AND EXISTS (
             SELECT 1 FROM public.lab_arena_runs AS partial_source
             WHERE partial_source.run_id = p_run_id
               AND partial_source.kind = 'score'
               AND pg_catalog.jsonb_typeof(partial_source.company_judgment_refs) = 'array'
           )
         ))$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_complete_attempt(text,text,jsonb,text,text)'::regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'partial_source.company_judgment_refs') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_partial_company_completion_patch_missing';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$partial_company_output$;

CREATE OR REPLACE FUNCTION public.lab_arena_complete_attempt_v3(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_result JSONB,
  p_terminal_cause TEXT,
  p_output_ref TEXT,
  p_output_hash TEXT,
  p_company_judgment_evidence JSONB,
  p_completion_request_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_complete_attempt_v3$
DECLARE
  v_run public.lab_arena_runs;
  v_result JSONB;
  v_context JSONB;
  v_item JSONB;
  v_doc JSONB;
  v_reservation public.lab_arena_company_judgment_reservations;
  v_expected INTEGER;
  v_supplied INTEGER;
BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs
  WHERE run_id = p_run_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_run_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_run.status IN ('accepted', 'failed') THEN
    RETURN public.lab_arena_complete_attempt(
      p_run_id, p_lease_token_hash, p_result, p_terminal_cause, p_output_ref
    );
  END IF;
  v_context := v_run.claim_response -> 'company_judgment_cache';
  IF v_run.kind <> 'score'
     OR pg_catalog.jsonb_typeof(v_run.company_judgment_refs)
        IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(v_context) IS DISTINCT FROM 'object'
     OR v_context ->> 'schema_version' IS DISTINCT FROM
        'leadpoet.lab_arena.company_judgment_lease.v1'
     OR pg_catalog.jsonb_typeof(p_company_judgment_evidence)
        IS DISTINCT FROM 'array'
     OR COALESCE(p_completion_request_hash, '')
        !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_completion_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF p_terminal_cause <> 'accepted' THEN
    IF pg_catalog.jsonb_array_length(p_company_judgment_evidence) = 0 THEN
      IF COALESCE(p_output_hash, '') <> '' OR COALESCE(p_output_ref, '') <> '' THEN
        RAISE EXCEPTION 'lab_arena_company_judgment_failure_invalid'
          USING ERRCODE = '22023';
      END IF;
      RETURN public.lab_arena_complete_attempt(
        p_run_id, p_lease_token_hash, p_result, p_terminal_cause, p_output_ref
      );
    END IF;
    IF p_terminal_cause NOT IN ('judge_error', 'judge_timeout', 'credential_error')
       OR pg_catalog.char_length(COALESCE(p_output_ref, '')) NOT BETWEEN 1 AND 1024 THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_failure_invalid'
        USING ERRCODE = '22023';
    END IF;
  END IF;
  IF COALESCE(p_output_hash, '') !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_output_hash_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.count(DISTINCT miss ->> 'cache_key')::INTEGER
  INTO v_expected
  FROM pg_catalog.jsonb_array_elements(v_context -> 'misses') AS misses(miss);
  SELECT pg_catalog.count(DISTINCT evidence ->> 'cache_key')::INTEGER
  INTO v_supplied
  FROM pg_catalog.jsonb_array_elements(
    p_company_judgment_evidence
  ) AS evidence_rows(evidence);
  IF (p_terminal_cause = 'accepted' AND v_expected <> v_supplied)
     OR v_supplied > v_expected
     OR v_supplied <> pg_catalog.jsonb_array_length(
       p_company_judgment_evidence
     ) THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_evidence_incomplete'
      USING ERRCODE = '22023';
  END IF;
  FOR v_item IN
    SELECT value
    FROM pg_catalog.jsonb_array_elements(p_company_judgment_evidence)
    ORDER BY value ->> 'cache_key'
  LOOP
    v_doc := v_item -> 'evidence_doc';
    IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
       OR COALESCE(v_item ->> 'cache_key', '')
          !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(v_item ->> 'company_input_hash', '')
          !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'authority_slot')
          IS DISTINCT FROM 'number'
       OR COALESCE(v_item ->> 'evidence_hash', '')
          !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_doc) IS DISTINCT FROM 'object'
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(v_context -> 'misses') AS miss(ref)
         WHERE miss.ref ->> 'cache_key' = v_item ->> 'cache_key'
           AND miss.ref ->> 'company_input_hash' =
               v_item ->> 'company_input_hash'
           AND (miss.ref ->> 'authority_slot')::INTEGER =
               (v_item ->> 'authority_slot')::INTEGER
       )
       OR v_doc ->> 'schema_version' IS DISTINCT FROM
          'leadpoet.lab_arena.company_judgment_evidence.v1'
       OR v_doc ->> 'cache_key' IS DISTINCT FROM v_item ->> 'cache_key'
       OR v_doc ->> 'company_input_hash' IS DISTINCT FROM
          v_item ->> 'company_input_hash'
       OR (v_doc ->> 'authority_slot')::INTEGER IS DISTINCT FROM
          (v_item ->> 'authority_slot')::INTEGER
       OR v_doc ->> 'source_score_run_id' IS DISTINCT FROM p_run_id
       OR v_doc ->> 'source_scored_run_id' IS DISTINCT FROM
          v_run.scored_run_id
       OR v_doc ->> 'source_output_ref' IS DISTINCT FROM p_output_ref
       OR v_doc ->> 'source_output_hash' IS DISTINCT FROM p_output_hash
       OR v_doc ->> 'source_runner_hotkey' IS DISTINCT FROM
          v_run.runner_hotkey
       OR v_doc ->> 'source_claim_request_id' IS DISTINCT FROM
          v_run.claim_request_id
       OR v_doc ->> 'source_claim_request_hash' IS DISTINCT FROM
          v_run.claim_request_hash
       OR (v_doc ->> 'source_lease_generation')::BIGINT IS DISTINCT FROM
          v_run.lease_generation
       OR v_doc ->> 'source_completion_request_hash' IS DISTINCT FROM
          p_completion_request_hash
       OR v_doc -> 'runner_authority_exclusions' IS DISTINCT FROM
          v_run.claim_response -> 'runner_authority_exclusions'
       OR pg_catalog.jsonb_typeof(v_doc -> 'raw_judgment')
          IS DISTINCT FROM 'object'
       OR (v_doc -> 'raw_judgment') ?| ARRAY[
         'company_index', 'company_qualified', 'duplicate_company',
         'duplicate_of_index'
       ] THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_evidence_invalid'
        USING ERRCODE = '22023';
    END IF;
    SELECT * INTO v_reservation
    FROM public.lab_arena_company_judgment_reservations
    WHERE cache_key = v_item ->> 'cache_key'
      AND authority_slot = (v_item ->> 'authority_slot')::INTEGER
      AND company_input_hash = v_item ->> 'company_input_hash'
      AND run_id = p_run_id
      AND lease_generation = v_run.lease_generation
      AND runner_hotkey = v_run.runner_hotkey
      AND lease_expires_at > pg_catalog.clock_timestamp()
    FOR UPDATE;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_reservation_stale'
        USING ERRCODE = 'P0003';
    END IF;
  END LOOP;

  v_result := public.lab_arena_complete_attempt(
    p_run_id, p_lease_token_hash,
    CASE WHEN p_terminal_cause <> 'accepted' THEN
      p_result || pg_catalog.jsonb_build_object(
        'company_judgment_completion_request_hash', p_completion_request_hash
      ) ELSE p_result END,
    p_terminal_cause, p_output_ref
  );
  IF v_result ->> 'status' IS DISTINCT FROM
       (CASE WHEN p_terminal_cause = 'accepted' THEN 'accepted' ELSE 'failed' END)
     OR v_result ->> 'idempotent' IS DISTINCT FROM 'false' THEN
    RETURN v_result;
  END IF;
  FOR v_item IN
    SELECT value
    FROM pg_catalog.jsonb_array_elements(p_company_judgment_evidence)
    ORDER BY value ->> 'cache_key'
  LOOP
    v_doc := v_item -> 'evidence_doc';
    INSERT INTO public.lab_arena_company_judgments (
      cache_key, authority_slot, scope_doc, company_input_hash,
      evidence_hash, evidence_doc, source_score_run_id,
      source_scored_run_id, source_runner_hotkey
    )
    SELECT
      v_item ->> 'cache_key',
      (v_item ->> 'authority_slot')::INTEGER,
      ref -> 'scope_doc',
      v_item ->> 'company_input_hash',
      v_item ->> 'evidence_hash',
      v_doc,
      p_run_id,
      v_run.scored_run_id,
      v_run.runner_hotkey
    FROM pg_catalog.jsonb_array_elements(
      v_run.company_judgment_refs
    ) AS company_refs(ref)
    WHERE ref ->> 'cache_key' = v_item ->> 'cache_key'
      AND ref ->> 'company_input_hash' = v_item ->> 'company_input_hash'
    ORDER BY (ref ->> 'company_index')::INTEGER
    LIMIT 1;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_ref_missing'
        USING ERRCODE = '22023';
    END IF;
  END LOOP;
  RETURN v_result || pg_catalog.jsonb_build_object(
    'company_judgments_stored', v_supplied,
    'company_judgments_reused',
      pg_catalog.jsonb_array_length(v_context -> 'hits')
  );
END;
$lab_arena_complete_attempt_v3$;
ALTER FUNCTION public.lab_arena_complete_attempt_v3(
  TEXT, TEXT, JSONB, TEXT, TEXT, TEXT, JSONB, TEXT
) OWNER TO lab_arena_owner;

-- An owner may replenish proved exhausted credit after a partial score.
-- Preserve every existing owner, stage, deadline, ledger and attempt gate.
-- Only the immutable cache rows written by the signed v3 completion can
-- establish the new nonempty failed-output alternative.
DO $partial_company_credit_retry$
DECLARE
  v_definition TEXT;
  v_old TEXT := 'AND previous.output_ref IS NULL';
  v_leader_old TEXT := 'v_group_leader := COALESCE(v_candidate.judgment_group_leader,FALSE);';
  v_leader_new TEXT := $leader$v_group_leader := CASE
      WHEN v_kind = 'score' AND v_candidate.company_judgment_refs IS NOT NULL
        AND v_candidate.judgment_cache_key IS NULL THEN NULL
      ELSE COALESCE(v_candidate.judgment_group_leader,FALSE) END;$leader$;
  v_new TEXT := $new$AND (
        previous.output_ref IS NULL
        OR (
          previous.kind = 'score'
          AND v_round.configuration_doc ->> 'company_quality_policy' = 'company_quality_v1'
          AND pg_catalog.jsonb_typeof(previous.company_judgment_refs) = 'array'
          AND previous.result_doc ->> 'company_judgment_completion_request_hash'
            ~ '^sha256:[0-9a-f]{64}$'
          AND EXISTS (
            SELECT 1 FROM public.lab_arena_company_judgments AS partial_credit_source
            WHERE partial_credit_source.source_score_run_id = previous.run_id
              AND partial_credit_source.source_scored_run_id = previous.scored_run_id
              AND partial_credit_source.source_runner_hotkey = previous.runner_hotkey
              AND partial_credit_source.evidence_doc ->> 'source_output_ref' = previous.output_ref
              AND partial_credit_source.evidence_doc ->> 'source_completion_request_hash' =
                  previous.result_doc ->> 'company_judgment_completion_request_hash'
              AND partial_credit_source.evidence_doc ->> 'source_claim_request_id' = previous.claim_request_id
              AND partial_credit_source.evidence_doc ->> 'source_claim_request_hash' = previous.claim_request_hash
              AND partial_credit_source.evidence_doc ->> 'source_lease_generation' = previous.lease_generation::TEXT
              AND partial_credit_source.evidence_doc -> 'runner_authority_exclusions' =
                  previous.claim_response -> 'runner_authority_exclusions'
              AND EXISTS (
                SELECT 1 FROM pg_catalog.jsonb_array_elements(previous.company_judgment_refs) AS refs(ref)
                WHERE ref ->> 'cache_key' = partial_credit_source.cache_key
                  AND ref ->> 'company_input_hash' = partial_credit_source.company_input_hash
                  AND ref -> 'scope_doc' = partial_credit_source.scope_doc
              )
          )
        )
      )$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
    'public.lab_arena_retry_credit_failures_v1(text,text,text,text)'
  ) IS NULL THEN
    RAISE EXCEPTION 'lab_arena_partial_company_credit_retry_requires_403';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_retry_credit_failures_v1(text,text,text,text)'::regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'partial_credit_source.source_score_run_id') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_partial_company_credit_retry_patch_missing';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
  -- Company-quality claims use per-company reservations rather than the
  -- legacy whole-item group. Keep their original NULL leader admission.
  IF pg_catalog.strpos(v_definition, v_leader_new) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_leader_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_partial_company_credit_leader_patch_missing';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_leader_old, v_leader_new);
  END IF;
  EXECUTE v_definition;
END;
$partial_company_credit_retry$;

NOTIFY pgrst, 'reload schema';
COMMIT;
