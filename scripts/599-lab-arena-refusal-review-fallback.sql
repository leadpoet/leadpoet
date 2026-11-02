-- Complete an explicit primary refusal with one independent full-source review.
-- Existing claims, ledger entries, decisions and published outcomes are unchanged.
-- DEPLOYMENT: drain review workers before applying this exact migration; deploy
-- the reviewer only after its fallback capability check succeeds. Existing
-- restart-guard locking and all admission/deadline/settlement gates are retained.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
LOCK TABLE public.lab_arena_rounds, public.lab_arena_submissions
  IN ACCESS EXCLUSIVE MODE;

DO $refusal599_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '38ed2f3616f92a0c17a3ef1c8f7617f229fed949f560f1bb3bb1f91c03c6534a';
  v_postimage CONSTANT TEXT := '2721c6f215655fc5ed3951e1a44082c367ead686248bd4fbbb548cdac4c74fcd';
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
    RAISE EXCEPTION 'Arena refusal599 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena refusal599 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old599_0_0$  IF v_submission.code_review_status = 'error' THEN$old599_0_0$;
  v_new:=$new599_0_0$  -- One alternate claim is durable in the existing reservation ledger. Check
  -- it before transient retry rules, including after an expired claim recovers.
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_ledger AS ledger
    WHERE ledger.submission_id = p_submission_id
      AND ledger.operation_id = 'openrouter.code_review'
      AND ledger.entry_kind = 'reservation'
      AND ledger.entry_doc ->> 'review_model' = 'openai/gpt-6.1-sol'
  ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'exhausted', 'code_review_status', v_submission.code_review_status,
      'attempt', v_submission.code_review_attempts
    );
  END IF;
  IF p_review_model = 'openai/gpt-6.1-sol' THEN
    IF v_submission.code_review_status <> 'error'
       OR v_submission.code_review_doc ->> 'error_code'
            IS DISTINCT FROM 'code_review_provider_refused'
       OR v_submission.code_review_doc ->> 'model'
            IS DISTINCT FROM 'anthropic/claude-sonnet-5.5' THEN
      RETURN pg_catalog.jsonb_build_object('status', 'fallback_not_authorized');
    END IF;
  ELSIF v_submission.code_review_doc ->> 'error_code' = 'code_review_provider_refused' THEN
    -- A stale worker cannot repeat the refused primary route.
    RETURN pg_catalog.jsonb_build_object('status', 'fallback_required');
  END IF;

  IF v_submission.code_review_status = 'error' THEN$new599_0_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_1$        'code_review_research_unverified'
      ) THEN TRUE$old599_0_1$;
  v_new:=$new599_0_1$        'code_review_research_unverified', 'code_review_provider_refused'
      ) THEN TRUE$new599_0_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_2$          'retryable', TRUE,
          'reserved_microusd'$old599_0_2$;
  v_new:=$new599_0_2$          'retryable', v_reservation.entry_doc ->> 'review_model' IS DISTINCT FROM 'openai/gpt-6.1-sol',
          'reserved_microusd'$new599_0_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_3$          'retryable', v_now < v_deadline AND v_round.status = 'open',$old599_0_3$;
  v_new:=$new599_0_3$          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND v_reservation.entry_doc ->> 'review_model' IS DISTINCT FROM 'openai/gpt-6.1-sol',$new599_0_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena refusal599 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena refusal599 readback differs' USING ERRCODE='55000';
  END IF;
END;
$refusal599_0$;

DO $refusal599_1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := 'cd09eb1c882c3be43b78415901eddb462feeb860b25e6affd7b86278c072d777';
  v_postimage CONSTANT TEXT := 'b11074d20dcc5dcef8ec76aedcf6a121da2b16ef3d2bccb902913917d239dddc';
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
    RAISE EXCEPTION 'Arena refusal599 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena refusal599 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old599_1_0$         'code_review_provider_request_rejected',$old599_1_0$;
  v_new:=$new599_1_0$         'code_review_provider_request_rejected', 'code_review_provider_refused',$new599_1_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_1$          'code_review_research_unverified', 'review_response_invalid'
        ))$old599_1_1$;
  v_new:=$new599_1_1$          'code_review_research_unverified', 'review_response_invalid',
          'code_review_provider_refused'
        ))$new599_1_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_2$       OR (v_retryable IS FALSE AND v_error_code IN ($old599_1_2$;
  v_new:=$new599_1_2$       OR (v_retryable IS FALSE
         AND p_review_doc ->> 'model' IS DISTINCT FROM 'openai/gpt-6.1-sol'
         AND v_error_code IN ($new599_1_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_3$  IF p_status = 'error' THEN$old599_1_3$;
  v_new:=$new599_1_3$  -- Every alternate failure is terminal. This is authoritative state, not
  -- the optional provider diagnostic projection. Binding is checked below.
  IF p_status = 'error' AND p_review_doc ->> 'model' = 'openai/gpt-6.1-sol' THEN
    p_review_doc := pg_catalog.jsonb_set(p_review_doc, '{retryable}', 'false'::JSONB);
  END IF;
  IF p_status = 'error' THEN$new599_1_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_4$                'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630')$old599_1_4$;
  v_new:=$new599_1_4$                'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630',
                'openai/gpt-6.1-sol','openai/gpt-6.1-sol-20260929')$new599_1_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_5$('Anthropic','Amazon Bedrock','Google','Google Vertex')$old599_1_5$;
  v_new:=$new599_1_5$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure')$new599_1_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_6$'endpoints_total','endpoint_shape','selected_count','selected_model',$old599_1_6$;
  v_new:=$new599_1_6$'endpoints_total','endpoint_shape','selected_count','selected_model','selected_provider',$new599_1_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena refusal599 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena refusal599 readback differs' USING ERRCODE='55000';
  END IF;
END;
$refusal599_1$;

DO $refusal599_2$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := '1461e5cd607007e9a59f659f6935b089e4ff80d872da82d5376b8e882b2992d2';
  v_postimage CONSTANT TEXT := '192414da9b28489c833e32ab35f9dbf881a65ff3dc81bac45fe15895ba3de15b';
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
    TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena refusal599 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena refusal599 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old599_2_0$    'atomic_restart_guard', TRUE,$old599_2_0$;
  v_new:=$new599_2_0$    'atomic_restart_guard', TRUE,
    'refusal_fallback', pg_catalog.jsonb_build_object(
      'primary_model', 'anthropic/claude-sonnet-5.5',
      'alternate_model', 'openai/gpt-6.1-sol',
      'max_alternate_claims', 1
    ),$new599_2_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 2.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena refusal599 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena refusal599 readback differs' USING ERRCODE='55000';
  END IF;
END;
$refusal599_2$;

NOTIFY pgrst, 'reload schema';
COMMIT;
