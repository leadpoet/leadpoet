-- Prospective DeepSeek admission primary; existing Sonnet/Sol episodes remain bound.
-- Function-only, append-only ledger semantics: no row/counter/deadline changes.
-- Requires exact642 cutoff and deployed637 recovery wrappers; preserves them.
-- Apply this additive database change before paired startup preflight.
-- The canonical controller then drains work before activating the new host.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE NOWAIT;

DO $requires637_643$
DECLARE target RECORD;
BEGIN
  FOR target IN SELECT * FROM (VALUES
    ('public.lab_arena__nov06_late_review635_v1(public.lab_arena_rounds,public.lab_arena_submissions)',
     '37f4106388a4683a297d46c470684dc17548f7431db6c72016cbae043836530f'),
    ('public.lab_arena_submissions_frozen_v1()',
     'a225ce8971679ea2d389789c7d3b940a871b202fff46a3859562efb8d49650cb')
  ) AS parents(signature,sha256) LOOP
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
         pg_catalog.to_regprocedure(target.signature)),'sha256'),'hex')
         IS DISTINCT FROM target.sha256 THEN
      RAISE EXCEPTION 'Arena primary643 parent637 differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$requires637_643$;

DO $primary643_0$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := 'd101aa5610e1c64aa92a975754a45cfdbfb70b1d7fff20832ddb1ea72dc148ff';
  v_postimage CONSTANT TEXT := '03009634a9dcde9342a7428d73eaf4285bf341811241a1d51e5d54bc8b8b04b9';
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
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena primary643 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena primary643 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old643_0_0$  v_required_model TEXT;$old643_0_0$;
  v_new:=$new643_0_0$  v_required_model TEXT;
  v_primary_model TEXT;$new643_0_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_1$AND reservation.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'$old643_0_1$;
  v_new:=$new643_0_1$AND reservation.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')$new643_0_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_2$SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'anthropic/claude-sonnet-5.5'),$old643_0_2$;
  v_new:=$new643_0_2$SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')),$new643_0_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_3$    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);$old643_0_3$;
  v_new:=$new643_0_3$    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  -- Freeze this episode to its first actual primary reservation. Historical
  -- Sonnet episodes keep Sonnet; model IDs and paid counters remain truthful.
  SELECT ledger.entry_doc ->> 'review_model' INTO v_primary_model
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND ledger.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start)
  ORDER BY ledger.entry_id LIMIT 1;
  v_primary_model := COALESCE(v_primary_model,'anthropic/claude-sonnet-5.5');$new643_0_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_4$THEN 'openai/gpt-6.1-sol' ELSE 'anthropic/claude-sonnet-5.5' END;$old643_0_4$;
  v_new:=$new643_0_4$THEN 'openai/gpt-6.1-sol' ELSE v_primary_model END;$new643_0_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_5$  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL$old643_0_5$;
  v_new:=$new643_0_5$  ELSIF NOT v_refusal_recovery
      AND v_reservation.entry_doc ->> 'review_model'='deepseek/deepseek-v4.1-flash-1017'
      AND v_head.entry_kind IN ('settlement','uncertain')
      AND v_head.entry_doc ->> 'error_code'='review_response_invalid'
      AND v_head.entry_doc ->> 'error_reason'='truncated' THEN
    -- Use the next existing invalid-review slot, never a 65536-token DS call.
    IF p_review_model IS DISTINCT FROM 'openai/gpt-6.1-sol' THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model','openai/gpt-6.1-sol');
    END IF;
  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL$new643_0_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_6$  ELSIF p_review_model='openai/gpt-6.1-sol' THEN
    RETURN pg_catalog.jsonb_build_object('status','fallback_not_authorized');
  END IF;$old643_0_6$;
  v_new:=$new643_0_6$  ELSIF p_review_model='openai/gpt-6.1-sol' THEN
    RETURN pg_catalog.jsonb_build_object('status','fallback_not_authorized');
  ELSIF v_primary_count>0 AND p_review_model IS DISTINCT FROM v_primary_model THEN
    RETURN pg_catalog.jsonb_build_object('status','route_required','review_model',v_primary_model);
  END IF;$new643_0_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_0_7$'review_alternate_attempts', v_alternate_count)$old643_0_7$;
  v_new:=$new643_0_7$'review_alternate_attempts', v_alternate_count, 'review_primary_model', v_primary_model)$new643_0_7$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 0.7 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena primary643 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena primary643 readback differs' USING ERRCODE='55000';
  END IF;
END;
$primary643_0$;

DO $primary643_1$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '167edcc637fc4d3c3ac5b624850788895a736cb88907e97172e43c077b609f6d';
  v_postimage CONSTANT TEXT := '91f4d574ab3d1d704228adac40507f6c4fb80a49fd091019ca713af1a0ff8a96';
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
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena primary643 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena primary643 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old643_1_0$  v_required_model TEXT;$old643_1_0$;
  v_new:=$new643_1_0$  v_required_model TEXT;
  v_primary_model TEXT;$new643_1_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_1_1$AND reservation.entry_doc ->> 'review_model'='anthropic/claude-sonnet-5.5'$old643_1_1$;
  v_new:=$new643_1_1$AND reservation.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')$new643_1_1$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.1 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_1_2$SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' = 'anthropic/claude-sonnet-5.5'),$old643_1_2$;
  v_new:=$new643_1_2$SELECT count(*) FILTER (WHERE ledger.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')),$new643_1_2$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.2 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_1_3$    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);$old643_1_3$;
  v_new:=$new643_1_3$    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start);
  -- Freeze this episode to its first actual primary reservation. Historical
  -- Sonnet episodes keep Sonnet; model IDs and paid counters remain truthful.
  SELECT ledger.entry_doc ->> 'review_model' INTO v_primary_model
  FROM public.lab_arena_ledger AS ledger
  WHERE ledger.submission_id=p_submission_id
    AND ledger.operation_id='openrouter.code_review' AND ledger.entry_kind='reservation'
    AND ledger.entry_doc ->> 'review_model' IN ('anthropic/claude-sonnet-5.5','deepseek/deepseek-v4.1-flash-1017')
    AND (v_episode_start IS NULL OR (ledger.entry_doc ->> 'attempt')::INTEGER>=v_episode_start)
  ORDER BY ledger.entry_id LIMIT 1;
  v_primary_model := COALESCE(v_primary_model,'anthropic/claude-sonnet-5.5');$new643_1_3$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.3 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_1_4$'openai/gpt-6.1-sol','openai/gpt-6.1-sol-20260929')$old643_1_4$;
  v_new:=$new643_1_4$'openai/gpt-6.1-sol','openai/gpt-6.1-sol-20260929',
                'deepseek/deepseek-v4.1-flash-1017','deepseek/deepseek-v4.1-flash-20261017')$new643_1_4$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.4 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_1_5$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure')$old643_1_5$;
  v_new:=$new643_1_5$('Anthropic','Amazon Bedrock','Google','Google Vertex','Azure','Nebius')$new643_1_5$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.5 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  v_old:=$old643_1_6$'review_alternate_attempts', v_alternate_count)$old643_1_6$;
  v_new:=$new643_1_6$'review_alternate_attempts', v_alternate_count, 'review_primary_model', v_primary_model)$new643_1_6$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 1.6 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena primary643 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena primary643 readback differs' USING ERRCODE='55000';
  END IF;
END;
$primary643_1$;

DO $primary643_2$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_code_review_schema_v1()';
  v_preimage CONSTANT TEXT := 'e2431ef03209fd2844a71ed80c068c9e289b087ecac7fa5e4a17b90bf436bda9';
  v_postimage CONSTANT TEXT := '55c3a3644ed53787bcff9af034e6be7b2a325474503e1bc5aa761de92a6d4d16';
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
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'s',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena primary643 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena primary643 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=v_definition;
  v_old:=$old643_2_0$    'atomic_restart_guard', TRUE,$old643_2_0$;
  v_new:=$new643_2_0$    'atomic_restart_guard', TRUE,
    'prospective_primary', pg_catalog.jsonb_build_object(
      'model', 'deepseek/deepseek-v4.1-flash-1017', 'legacy_primary_model', 'anthropic/claude-sonnet-5.5',
      'episode_binding', 'first_primary_reservation', 'max_output_tokens', 32768,
      'truncation_alternate', 'openai/gpt-6.1-sol'),$new643_2_0$;
  IF length(v_updated)-length(pg_catalog.replace(v_updated,v_old,''))<>length(v_old)*1 THEN
    RAISE EXCEPTION 'Arena primary643 anchor 2.0 differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_updated,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena primary643 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena primary643 readback differs' USING ERRCODE='55000';
  END IF;
END;
$primary643_2$;

COMMIT;
