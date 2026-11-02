-- Bound explicit refusal recovery to two primary and two alternate reservations.
-- Legacy recovery requires current-attempt structured ledger proof, never prose.
-- No historical rows or ledger records are rewritten. Drain workers before apply.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
LOCK TABLE public.lab_arena_rounds, public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;

DO $refusal599_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '38ed2f3616f92a0c17a3ef1c8f7617f229fed949f560f1bb3bb1f91c03c6534a';
  v_postimage CONSTANT TEXT := '4b6560038d304138cccd174ed080313328c098a399bc8d50042099bc6d21986b';
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
  v_old:=$old599_0_0$  v_round_id TEXT;$old599_0_0$;
  v_new:=$new599_0_0$  v_episode_start INTEGER;
  v_primary_count INTEGER;
  v_alternate_count INTEGER;
  v_refusal_recovery BOOLEAN := FALSE;
  v_legacy_refusal BOOLEAN := FALSE;
  v_required_model TEXT;
  v_round_id TEXT;$new599_0_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_1$  IF v_submission.code_review_status = 'reviewing' THEN$old599_0_1$;
  v_new:=$new599_0_1$  IF v_submission.code_review_attempts>0 THEN
    v_call_identity := 'sha256:' || pg_catalog.encode(extensions.digest(
      pg_catalog.convert_to('lab_arena.code_review:' || p_submission_id || ':' ||
        v_submission.code_review_attempts::TEXT,'UTF8'),'sha256'),'hex');
    v_head := public.lab_arena__ledger_head(v_call_identity);
    SELECT * INTO v_reservation FROM public.lab_arena_ledger
      WHERE call_identity=v_call_identity AND entry_kind='reservation';
    v_legacy_refusal := COALESCE(
      v_submission.code_review_status='error'
      AND v_submission.code_review_doc ->> 'error_code'='code_review_provider_request_rejected'
      AND v_submission.code_review_doc -> 'provider_http_status'='200'::JSONB
      AND v_head.entry_kind IN ('settlement','uncertain')
      AND v_head.entry_doc ->> 'error_code'='code_review_provider_request_rejected'
      AND v_head.entry_doc -> 'provider_http_status'='200'::JSONB
      AND v_reservation.entry_doc -> 'attempt'=pg_catalog.to_jsonb(v_submission.code_review_attempts)
      AND v_reservation.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'
      AND TRUE,
      FALSE);
    v_refusal_recovery := COALESCE(v_reservation.entry_doc -> 'refusal_recovery'='true'::JSONB,FALSE)
      OR v_legacy_refusal
      OR (v_submission.code_review_status='error'
        AND v_submission.code_review_doc ->> 'error_code'='code_review_provider_refused'
        AND v_head.entry_kind IN ('settlement','uncertain')
        AND v_head.entry_doc ->> 'error_code'='code_review_provider_refused');
  END IF;
  IF v_refusal_recovery THEN
    v_episode_start := (v_reservation.entry_doc ->> 'refusal_episode_start')::INTEGER;
    IF v_episode_start IS NULL THEN
      SELECT min((reservation.entry_doc ->> 'attempt')::INTEGER) INTO v_episode_start
      FROM public.lab_arena_ledger reservation JOIN public.lab_arena_ledger terminal
        ON terminal.call_identity=reservation.call_identity
      WHERE reservation.submission_id=p_submission_id
        AND reservation.operation_id='openrouter.code_review' AND reservation.entry_kind='reservation'
        AND reservation.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'
        AND terminal.entry_kind IN ('settlement','uncertain')
        AND (terminal.entry_doc ->> 'error_code'='code_review_provider_refused'
          OR (terminal.entry_doc ->> 'error_code'='code_review_provider_request_rejected'
              AND terminal.entry_doc -> 'provider_http_status'='200'::JSONB));
    END IF;
    v_episode_start := COALESCE(v_episode_start,v_submission.code_review_attempts);
  END IF;
  SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'anthropic/claude-sonnet-5.5'),
         count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'openai/gpt-6.1-sol')
  INTO v_primary_count,v_alternate_count
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  IF v_submission.code_review_status = 'reviewing' THEN$new599_0_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_2$  IF v_submission.code_review_status = 'error' THEN$old599_0_2$;
  v_new:=$new599_0_2$  IF v_refusal_recovery AND (v_submission.code_review_doc ->> 'error_code' IN (
      'code_review_provider_refused','code_review_preparation_unavailable',
      'code_review_provider_rate_limited','code_review_provider_timeout',
      'code_review_provider_unavailable','code_review_transport_error',
      'review_response_invalid','code_review_research_unverified')
      OR v_legacy_refusal) THEN
    IF v_alternate_count>=2 THEN
      RETURN pg_catalog.jsonb_build_object('status','exhausted','code_review_status',v_submission.code_review_status,
        'attempt',v_submission.code_review_attempts);
    END IF;
    v_required_model := CASE WHEN v_primary_count>=2 OR v_alternate_count>0
      THEN 'openai/gpt-6.1-sol' ELSE 'anthropic/claude-sonnet-5.5' END;
    IF p_review_model IS DISTINCT FROM v_required_model THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model',v_required_model);
    END IF;
  ELSIF p_review_model='openai/gpt-6.1-sol' THEN
    RETURN pg_catalog.jsonb_build_object('status','fallback_not_authorized');
  END IF;

  IF v_submission.code_review_status = 'error' THEN$new599_0_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_3$    v_retryable := CASE$old599_0_3$;
  v_new:=$new599_0_3$    v_retryable := CASE
      WHEN v_refusal_recovery AND (v_submission.code_review_doc ->> 'error_code' IN (
      'code_review_provider_refused','code_review_preparation_unavailable',
      'code_review_provider_rate_limited','code_review_provider_timeout',
      'code_review_provider_unavailable','code_review_transport_error',
      'review_response_invalid','code_review_research_unverified')
      OR v_legacy_refusal) THEN TRUE$new599_0_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_4$      'review_model', p_review_model,$old599_0_4$;
  v_new:=$new599_0_4$      'review_model', p_review_model, 'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start,$new599_0_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_5$          'retryable', TRUE,
          'reserved_microusd'$old599_0_5$;
  v_new:=$new599_0_5$          'retryable', NOT v_refusal_recovery OR v_alternate_count<2,
          'reserved_microusd'$new599_0_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_6$          'retryable', v_now < v_deadline AND v_round.status = 'open',$old599_0_6$;
  v_new:=$new599_0_6$          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND (NOT v_refusal_recovery OR v_alternate_count<2),$new599_0_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_0_7$          'review_cost_microusd', v_review_cost
        ),$old599_0_7$;
  v_new:=$new599_0_7$          'review_cost_microusd', v_review_cost
        ) || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count) ELSE '{}'::JSONB END,$new599_0_7$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 0.7 differs' USING ERRCODE='55000';
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
  v_postimage CONSTANT TEXT := '881ed3c299f5989d1e481416ec645ba2686ab646596a4aae4eb0f2aa2807e8a6';
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
  v_old:=$old599_1_0$  v_round_id TEXT;$old599_1_0$;
  v_new:=$new599_1_0$  v_episode_start INTEGER;
  v_primary_count INTEGER;
  v_alternate_count INTEGER;
  v_refusal_recovery BOOLEAN := FALSE;
  v_legacy_refusal BOOLEAN := FALSE;
  v_required_model TEXT;
  v_round_id TEXT;$new599_1_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_1$         'code_review_provider_request_rejected',$old599_1_1$;
  v_new:=$new599_1_1$         'code_review_provider_request_rejected', 'code_review_provider_refused',$new599_1_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_2$          'code_review_research_unverified', 'review_response_invalid'
        ))$old599_1_2$;
  v_new:=$new599_1_2$          'code_review_research_unverified', 'review_response_invalid',
          'code_review_provider_refused'
        ))$new599_1_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_3$       OR (v_retryable IS FALSE AND v_error_code IN ($old599_1_3$;
  v_new:=$new599_1_3$       OR (v_retryable IS FALSE
         AND p_review_doc ->> 'model' IS DISTINCT FROM 'openai/gpt-6.1-sol'
         AND v_error_code IN ($new599_1_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_4$  IF v_head.entry_kind <> 'dispatch' OR v_reservation.entry_id IS NULL THEN$old599_1_4$;
  v_new:=$new599_1_4$  v_refusal_recovery := COALESCE(v_reservation.entry_doc -> 'refusal_recovery'='true'::JSONB,FALSE)
    OR v_error_code='code_review_provider_refused';
  IF v_refusal_recovery THEN
    v_episode_start := (v_reservation.entry_doc ->> 'refusal_episode_start')::INTEGER;
    IF v_episode_start IS NULL THEN
      SELECT min((reservation.entry_doc ->> 'attempt')::INTEGER) INTO v_episode_start
      FROM public.lab_arena_ledger reservation JOIN public.lab_arena_ledger terminal
        ON terminal.call_identity=reservation.call_identity
      WHERE reservation.submission_id=p_submission_id
        AND reservation.operation_id='openrouter.code_review' AND reservation.entry_kind='reservation'
        AND reservation.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'
        AND terminal.entry_kind IN ('settlement','uncertain')
        AND (terminal.entry_doc ->> 'error_code'='code_review_provider_refused'
          OR (terminal.entry_doc ->> 'error_code'='code_review_provider_request_rejected'
              AND terminal.entry_doc -> 'provider_http_status'='200'::JSONB));
    END IF;
    v_episode_start := COALESCE(v_episode_start,v_submission.code_review_attempts);
  END IF;
  SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'anthropic/claude-sonnet-5.5'),
         count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'openai/gpt-6.1-sol')
  INTO v_primary_count,v_alternate_count
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  IF v_refusal_recovery AND p_status='error' THEN
    v_retryable := v_error_code IN ('code_review_provider_refused','code_review_preparation_unavailable',
      'code_review_provider_rate_limited','code_review_provider_timeout','code_review_provider_unavailable',
      'code_review_transport_error','review_response_invalid','code_review_research_unverified')
      AND v_alternate_count<2;
    v_diagnostic := v_diagnostic || pg_catalog.jsonb_build_object('retryable',v_retryable);
  END IF;
  IF v_head.entry_kind <> 'dispatch' OR v_reservation.entry_id IS NULL THEN$new599_1_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_5$  END || v_research_doc;$old599_1_5$;
  v_new:=$new599_1_5$  END || v_research_doc || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count) ELSE '{}'::JSONB END;$new599_1_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_6$    'code_review_status', v_effective_status,$old599_1_6$;
  v_new:=$new599_1_6$    'code_review_status', v_effective_status,
    'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start, 'retryable', v_retryable,
    'review_primary_attempts', v_primary_count, 'review_alternate_attempts', v_alternate_count,$new599_1_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_7$                'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630')$old599_1_7$;
  v_new:=$new599_1_7$                'anthropic/claude-sonnet-5','anthropic/claude-sonnet-5-20260630',
                'openai/gpt-6.1-sol','openai/gpt-6.1-sol-20260929')$new599_1_7$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.7 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_8$('Anthropic','Amazon Bedrock','Google','Google Vertex')$old599_1_8$;
  v_new:=$new599_1_8$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure')$new599_1_8$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.8 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old599_1_9$'endpoints_total','endpoint_shape','selected_count','selected_model',$old599_1_9$;
  v_new:=$new599_1_9$'endpoints_total','endpoint_shape','selected_count','selected_model','selected_provider',$new599_1_9$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena refusal599 anchor 1.9 differs' USING ERRCODE='55000';
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
  v_postimage CONSTANT TEXT := 'cdad90e7b85438baf649b37bf8051b1d1db37244c426d5f46e6ab3d0eb6bfcd8';
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
      'max_primary_refusal_claims', 2, 'max_alternate_claims', 2,
      'legacy_structured_refusal_recovery', TRUE
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
