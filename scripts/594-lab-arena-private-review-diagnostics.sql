-- Keep bounded reviewer outcome observations only in existing private per-attempt
-- ledger entries. No historical rows, public projections or decisions change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $private_review594$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '90f66612d21b2cde658c4b9107bc63025eae9a96a3872ca5dbb9c18da60cdb20';
  v_postimage CONSTANT TEXT := '26c5f4f7d473c9b509823da99a9d0a4bd4b619a8690cf0ded9f08db00545eb9f';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_old TEXT;
  v_new TEXT;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_metadata,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena review594 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review594 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old594_0$  v_retryable BOOLEAN;$old594_0$;
  v_new:=$new594_0$  v_retryable BOOLEAN;
  v_provider_doc JSONB := '{}'::JSONB;
  v_provider_input JSONB;
  v_provider_field TEXT;
  v_provider_value JSONB;$new594_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review594 anchor 0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old594_1$  IF p_status = 'error' THEN$old594_1$;
  v_new:=$new594_1$  -- A private observation is not verdict, cost or retry authority.
  IF p_review_doc ? 'provider_diagnostic' THEN
    v_provider_input:=p_review_doc->'provider_diagnostic';
    IF pg_catalog.jsonb_typeof(v_provider_input) IS DISTINCT FROM 'object'
       OR pg_catalog.octet_length(v_provider_input::TEXT)>2048 THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    IF NOT v_provider_input ?& ARRAY[
      'finish_reason','native_finish_reason','max_output_tokens','context_integrity'] THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    FOR v_provider_field,v_provider_value IN
      SELECT key,value FROM pg_catalog.jsonb_each(v_provider_input)
    LOOP
      IF v_provider_field IN ('max_output_tokens','prompt_tokens','completion_tokens','reasoning_tokens') THEN
        IF pg_catalog.jsonb_typeof(v_provider_value) IS DISTINCT FROM 'number'
           OR v_provider_value::TEXT !~ '^(0|[1-9][0-9]{0,9})$' THEN
          RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
        END IF;
        IF v_provider_value::TEXT::BIGINT>1000000000
           OR (v_provider_field='max_output_tokens' AND v_provider_value='0'::JSONB) THEN
          RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
        END IF;
      ELSE
        IF pg_catalog.jsonb_typeof(v_provider_value) IS DISTINCT FROM 'string' THEN
          RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
        END IF;
        IF (CASE
          WHEN v_provider_field IN ('finish_reason','native_finish_reason') THEN
            v_provider_value #>> '{}' NOT IN (
              'stop','length','tool_calls','function_call','content_filter','error',
              'end_turn','max_tokens','stop_sequence','tool_use','refusal',
              'cancelled','completed','missing','invalid','other')
          WHEN v_provider_field='context_integrity' THEN
            v_provider_value #>> '{}' NOT IN (
              'not_required','verified','metadata_missing','requested_model','strategy',
              'attempt','pipeline','endpoints_missing','endpoints_available',
              'endpoints_total','endpoint_shape','selected_count','selected_model',
              'usage_missing','prompt_tokens','context_budget')
          WHEN v_provider_field IN ('response_model','selected_model') THEN
            v_provider_value #>> '{}' NOT IN (
              'anthropic/claude-sonnet-5.5','anthropic/claude-sonnet-5.5-20260928',
              'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630')
          WHEN v_provider_field='selected_provider' THEN
            v_provider_value #>> '{}' NOT IN ('Anthropic','Amazon Bedrock','Google','Google Vertex')
          WHEN v_provider_field='generation_id' THEN
            v_provider_value #>> '{}' !~ '^gen-[0-9]{1,20}-[A-Za-z0-9_-]{1,128}$'
          ELSE TRUE
        END) THEN
          RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
        END IF;
      END IF;
    END LOOP;
    v_provider_doc:=pg_catalog.jsonb_build_object('provider_diagnostic',v_provider_input);
  END IF;
  IF p_status = 'error' THEN$new594_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review594 anchor 1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old594_2$        'reserved_microusd', v_reservation.amount_microusd
      ) || v_diagnostic$old594_2$;
  v_new:=$new594_2$        'reserved_microusd', v_reservation.amount_microusd
      ) || v_diagnostic || v_provider_doc$new594_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review594 anchor 2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old594_3$        'review_status', v_effective_status
      ) || v_diagnostic,$old594_3$;
  v_new:=$new594_3$        'review_status', v_effective_status
      ) || v_diagnostic || v_provider_doc,$new594_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena review594 anchor 3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review594 postimage differs: %', pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex') USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review594 readback differs' USING ERRCODE='55000';
  END IF;
END;
$private_review594$;

COMMIT;
