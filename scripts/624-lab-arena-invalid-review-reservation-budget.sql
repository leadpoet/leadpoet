-- Bound invalid/unverified reviews to three durable reservations from the first
-- unusable outcome; an untagged legacy error starts at its latest bound failure.
-- Preserve refusal2+2 ceilings, prior costs, deadlines and final rows.
-- No historical writes, new RPCs, tables, roles or diagnostic policy authority.
-- Apply only through the canonical drained-release flow.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
LOCK TABLE public.lab_arena_rounds, public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;

DO $invalid624_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '4b6560038d304138cccd174ed080313328c098a399bc8d50042099bc6d21986b';
  v_postimage CONSTANT TEXT := 'd057d4c76e4b26c4251673aa6b0aa2d7e1069e177ddcc12ecb0b94c4f1c95bf2';
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
    RAISE EXCEPTION 'Arena invalid624 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena invalid624 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old624_0_0$DECLARE
  v_review_guard_present BOOLEAN;
  v_episode_start INTEGER;
  v_primary_count INTEGER;
$old624_0_0$;
  v_new:=$new624_0_0$DECLARE
  v_review_guard_present BOOLEAN;
  v_output_token_recommendation INTEGER := 32768;
  v_invalid_start INTEGER;
  v_invalid_count INTEGER;
  v_invalid_doc JSONB := '{}'::JSONB;
  v_episode_start INTEGER;
  v_primary_count INTEGER;
$new624_0_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_1$    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  IF v_submission.code_review_status = 'reviewing' THEN
    IF v_submission.code_review_claim = p_claim_token_hash THEN
$old624_0_1$;
  v_new:=$new624_0_1$    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  v_output_token_recommendation := COALESCE(
    (v_reservation.entry_doc ->> 'review_output_token_recommendation')::INTEGER,32768);
  IF v_head.entry_kind IN ('settlement','uncertain')
     AND v_head.entry_doc ->> 'error_code'='review_response_invalid'
     AND v_head.entry_doc ->> 'error_reason' IN ('truncated','truncated_at_limit') THEN
    v_output_token_recommendation := 65536;
  END IF;
  -- A new finite episode starts at the latest bound unusable review, not at
  -- the global attempt count or at historical unrelated infrastructure errors.
  v_invalid_start := (v_reservation.entry_doc ->> 'invalid_episode_start')::INTEGER;
  IF v_invalid_start IS NULL AND v_submission.code_review_status='error'
     AND v_reservation.entry_id IS NOT NULL
     AND v_reservation.entry_doc -> 'attempt'=pg_catalog.to_jsonb(v_submission.code_review_attempts)
     AND v_head.entry_kind IN ('settlement','uncertain')
     AND v_head.entry_doc ->> 'error_code' IN
       ('review_response_invalid','code_review_research_unverified')
     AND v_head.entry_doc ->> 'error_code' IS NOT DISTINCT FROM
       v_submission.code_review_doc ->> 'error_code' THEN
    v_invalid_start := v_submission.code_review_attempts;
  END IF;
  IF v_invalid_start IS NOT NULL THEN
    SELECT count(*) INTO v_invalid_count FROM public.lab_arena_ledger ledger
    WHERE ledger.submission_id=p_submission_id
      AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
      AND (ledger.entry_doc ->> 'attempt')::INTEGER>=v_invalid_start;
    v_invalid_doc := pg_catalog.jsonb_build_object(
      'invalid_episode_start',v_invalid_start,'review_invalid_attempts',v_invalid_count);
  END IF;
  IF v_submission.code_review_status = 'reviewing' THEN
    IF v_submission.code_review_claim = p_claim_token_hash THEN
$new624_0_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_2$          'reason', 'code_review_claim_expired',
          'error_code', 'code_review_transport_error',
          'retryable', NOT v_refusal_recovery OR v_alternate_count<2,
          'reserved_microusd', v_reservation.amount_microusd
        )
$old624_0_2$;
  v_new:=$new624_0_2$          'reason', 'code_review_claim_expired',
          'error_code', 'code_review_transport_error',
          'retryable', (NOT v_refusal_recovery OR v_alternate_count<2)
            AND (v_invalid_start IS NULL OR v_invalid_count<3),
          'reserved_microusd', v_reservation.amount_microusd
        )
$new624_0_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_3$          END,
          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND (NOT v_refusal_recovery OR v_alternate_count<2),
          'model', v_reservation.entry_doc ->> 'review_model',
          'file_count', v_reservation.entry_doc -> 'file_count',
$old624_0_3$;
  v_new:=$new624_0_3$          END,
          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND (NOT v_refusal_recovery OR v_alternate_count<2)
            AND (v_invalid_start IS NULL OR v_invalid_count<3),
          'model', v_reservation.entry_doc ->> 'review_model',
          'file_count', v_reservation.entry_doc -> 'file_count',
$new624_0_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_4$        ) || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count) ELSE '{}'::JSONB END,
        code_review_started_at = v_now,
        code_review_expires_at = NULL
$old624_0_4$;
  v_new:=$new624_0_4$        ) || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count) ELSE '{}'::JSONB END || v_invalid_doc || pg_catalog.jsonb_build_object(
          'review_output_token_recommendation',v_output_token_recommendation),
        code_review_started_at = v_now,
        code_review_expires_at = NULL
$new624_0_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_5$      'retry_after', v_freeze_at
    );
  END IF;

$old624_0_5$;
  v_new:=$new624_0_5$      'retry_after', v_freeze_at
    );
  END IF;

  -- Always recover/settle an expired lease above before enforcing the cap.
  -- Refusal does not reset an existing invalid-review episode.
  IF (v_invalid_start IS NOT NULL AND v_invalid_count>=3)
     OR (v_submission.code_review_status='error'
       AND v_head.entry_kind IN ('settlement','uncertain')
       AND v_head.entry_doc ->> 'error_code'='review_response_invalid'
       AND v_head.entry_doc ->> 'error_reason'='truncated_at_limit') THEN
    RETURN pg_catalog.jsonb_build_object('status','exhausted','code_review_status',v_submission.code_review_status,
      'attempt',v_submission.code_review_attempts);
  END IF;

$new624_0_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_6$      'attempt', v_attempt,
      'review_model', p_review_model, 'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start,
      'file_count', p_file_count,
      'source_bytes', p_source_bytes
$old624_0_6$;
  v_new:=$new624_0_6$      'attempt', v_attempt,
      'review_model', p_review_model, 'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start,
      'invalid_episode_start', v_invalid_start,
      'review_output_token_recommendation',v_output_token_recommendation,
      'file_count', p_file_count,
      'source_bytes', p_source_bytes
$new624_0_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_0_7$    'claim_expires_at', v_expires,
    'call_identity', v_call_identity,
    'reserved_microusd', p_reservation_microusd
  );
END;
$old624_0_7$;
  v_new:=$new624_0_7$    'claim_expires_at', v_expires,
    'call_identity', v_call_identity,
    'reserved_microusd', p_reservation_microusd,
    'review_output_token_recommendation',v_output_token_recommendation
  );
END;
$new624_0_7$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 0.7 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena invalid624 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena invalid624 readback differs' USING ERRCODE='55000';
  END IF;
END;
$invalid624_0$;

DO $invalid624_1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '881ed3c299f5989d1e481416ec645ba2686ab646596a4aae4eb0f2aa2807e8a6';
  v_postimage CONSTANT TEXT := '2e273f7683e33c4749996951919cfc229a74c3269d860c58980b0be211bcb78b';
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
    RAISE EXCEPTION 'Arena invalid624 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena invalid624 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old624_1_0$AS $function$
DECLARE
  v_episode_start INTEGER;
  v_primary_count INTEGER;
$old624_1_0$;
  v_new:=$new624_1_0$AS $function$
DECLARE
  v_output_token_recommendation INTEGER := 32768;
  v_invalid_start INTEGER;
  v_invalid_count INTEGER;
  v_invalid_doc JSONB := '{}'::JSONB;
  v_episode_start INTEGER;
  v_primary_count INTEGER;
$new624_1_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 1.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_1_1$           pg_catalog.jsonb_typeof(p_review_doc -> 'error_reason') <> 'string'
           OR p_review_doc ->> 'error_reason' NOT IN (
             'envelope', 'model_mismatch', 'choice', 'not_finished',
             'message', 'refusal_or_tools', 'content', 'content_json',
             'document_keys', 'verdict', 'summary', 'coverage',
$old624_1_1$;
  v_new:=$new624_1_1$           pg_catalog.jsonb_typeof(p_review_doc -> 'error_reason') <> 'string'
           OR p_review_doc ->> 'error_reason' NOT IN (
             'envelope', 'model_mismatch', 'choice', 'not_finished', 'truncated', 'truncated_at_limit',
             'message', 'refusal_or_tools', 'content', 'content_json',
             'document_keys', 'verdict', 'summary', 'coverage',
$new624_1_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 1.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_1_2$    v_diagnostic := v_diagnostic || pg_catalog.jsonb_build_object('retryable',v_retryable);
  END IF;
  IF v_head.entry_kind <> 'dispatch' OR v_reservation.entry_id IS NULL THEN
    RAISE EXCEPTION 'lab_arena_code_review_ledger_invalid'
$old624_1_2$;
  v_new:=$new624_1_2$    v_diagnostic := v_diagnostic || pg_catalog.jsonb_build_object('retryable',v_retryable);
  END IF;
  v_output_token_recommendation := COALESCE(
    (v_reservation.entry_doc ->> 'review_output_token_recommendation')::INTEGER,32768);
  IF v_error_code='review_response_invalid'
     AND p_review_doc ->> 'error_reason' IN ('truncated','truncated_at_limit') THEN
    v_output_token_recommendation := 65536;
  END IF;
  -- Reservation-owned state survives transport errors, expiry and refusals.
  -- Ignore caller-supplied counters and optional provider observations.
  v_invalid_start := (v_reservation.entry_doc ->> 'invalid_episode_start')::INTEGER;
  IF v_invalid_start IS NULL AND p_status='error'
     AND v_error_code IN ('review_response_invalid','code_review_research_unverified') THEN
    v_invalid_start := v_submission.code_review_attempts;
  END IF;
  IF v_invalid_start IS NOT NULL THEN
    SELECT count(*) INTO v_invalid_count FROM public.lab_arena_ledger ledger
    WHERE ledger.submission_id=p_submission_id
      AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
      AND (ledger.entry_doc ->> 'attempt')::INTEGER>=v_invalid_start;
    v_invalid_doc := pg_catalog.jsonb_build_object(
      'invalid_episode_start',v_invalid_start,'review_invalid_attempts',v_invalid_count);
    IF p_status='error' THEN
      v_retryable := v_error_code IN ('code_review_provider_refused','code_review_preparation_unavailable',
        'code_review_provider_rate_limited','code_review_provider_timeout','code_review_provider_unavailable',
        'code_review_transport_error','review_response_invalid','code_review_research_unverified')
        AND v_invalid_count<3 AND (NOT v_refusal_recovery OR v_alternate_count<2)
        AND NOT (v_error_code='review_response_invalid'
          AND COALESCE(p_review_doc ->> 'error_reason'='truncated_at_limit',FALSE));
      v_diagnostic := v_diagnostic || pg_catalog.jsonb_build_object('retryable',v_retryable);
    END IF;
  END IF;
  IF v_head.entry_kind <> 'dispatch' OR v_reservation.entry_id IS NULL THEN
    RAISE EXCEPTION 'lab_arena_code_review_ledger_invalid'
$new624_1_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 1.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_1_3$  END || v_research_doc || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count) ELSE '{}'::JSONB END;
  -- Recheck after both locks. A provider result can settle after the deadline,
  -- but it cannot create a new participant-admissible pass.
$old624_1_3$;
  v_new:=$new624_1_3$  END || v_research_doc || CASE WHEN v_refusal_recovery THEN pg_catalog.jsonb_build_object(
      'refusal_recovery', TRUE, 'refusal_episode_start', v_episode_start, 'review_primary_attempts', v_primary_count,
      'review_alternate_attempts', v_alternate_count) ELSE '{}'::JSONB END || v_invalid_doc || pg_catalog.jsonb_build_object(
      'review_output_token_recommendation',v_output_token_recommendation);
  -- Recheck after both locks. A provider result can settle after the deadline,
  -- but it cannot create a new participant-admissible pass.
$new624_1_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 1.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old624_1_4$    'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start, 'retryable', v_retryable,
    'review_primary_attempts', v_primary_count, 'review_alternate_attempts', v_alternate_count,
    'attempt', v_submission.code_review_attempts,
    'ledger_status', v_ledger_status,
$old624_1_4$;
  v_new:=$new624_1_4$    'refusal_recovery', v_refusal_recovery, 'refusal_episode_start', v_episode_start, 'retryable', v_retryable,
    'review_primary_attempts', v_primary_count, 'review_alternate_attempts', v_alternate_count,
    'invalid_episode_start', v_invalid_start, 'review_invalid_attempts', v_invalid_count,
    'review_output_token_recommendation',v_output_token_recommendation,
    'attempt', v_submission.code_review_attempts,
    'ledger_status', v_ledger_status,
$new624_1_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 1.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena invalid624 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena invalid624 readback differs' USING ERRCODE='55000';
  END IF;
END;
$invalid624_1$;

DO $invalid624_2$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := 'cdad90e7b85438baf649b37bf8051b1d1db37244c426d5f46e6ab3d0eb6bfcd8';
  v_postimage CONSTANT TEXT := 'e2431ef03209fd2844a71ed80c068c9e289b087ecac7fa5e4a17b90bf436bda9';
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
    RAISE EXCEPTION 'Arena invalid624 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena invalid624 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old624_2_0$      'legacy_structured_refusal_recovery', TRUE
    ),
    'claim_ttl_seconds', 600,
    'retry_backoff_seconds', 60,
$old624_2_0$;
  v_new:=$new624_2_0$      'legacy_structured_refusal_recovery', TRUE
    ),
    'invalid_review_budget', pg_catalog.jsonb_build_object(
      'max_reservations',3, 'starts_at','first_invalid_or_unverified_result',
      'legacy_untagged_origin','latest_bound_invalid_or_unverified_result',
      'counts_expired_and_transport_claims',TRUE, 'refusal_does_not_reset',TRUE,
      'output_tokens',pg_catalog.jsonb_build_array(32768,65536), 'truncated_at_limit_terminal',TRUE
    ),
    'claim_ttl_seconds', 600,
    'retry_backoff_seconds', 60,
$new624_2_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena invalid624 anchor 2.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena invalid624 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena invalid624 readback differs' USING ERRCODE='55000';
  END IF;
END;
$invalid624_2$;

NOTIFY pgrst, 'reload schema';
COMMIT;
