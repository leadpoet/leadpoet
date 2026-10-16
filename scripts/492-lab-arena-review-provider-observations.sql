-- Retain bounded OpenRouter review observations in the existing private docs.
-- No historical rows, retry decisions, costs or participant outcomes change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $review_provider_observations492$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)';
  v_preimage CONSTANT TEXT := '30a779b2bb4eec20652282a36f7a5abd1108df46667867d0335ac112283882df';
  v_postimage CONSTANT TEXT := '7db9233a48f24be3cf2fa61572546774cccefc1d69d95af2ad3c5dc10bea8083';
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
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
    RAISE EXCEPTION 'Arena review492 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena review492 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,$old492a$    v_http_status := (p_review_doc ->> 'provider_http_status')::INTEGER;$old492a$,
    $new492a$    -- Optional private observations never change retry or accounting authority.
    IF EXISTS (
      SELECT 1 FROM pg_catalog.jsonb_each(p_review_doc) AS observation(name,value)
      WHERE observation.name IN ('provider_raw_http_status','provider_effective_http_status')
        AND (pg_catalog.jsonb_typeof(observation.value) IS DISTINCT FROM 'number'
          OR observation.value::TEXT !~ '^[1-5][0-9]{2}$')
    ) OR (p_review_doc ? 'provider_limit_source' AND (
      pg_catalog.jsonb_typeof(p_review_doc->'provider_limit_source') IS DISTINCT FROM 'string'
      OR p_review_doc->>'provider_limit_source' NOT IN
        ('openrouter_credits','openrouter_key_limit','openrouter_in_flight_budget','unknown')
    )) THEN
      RAISE EXCEPTION 'lab_arena_code_review_input_invalid' USING ERRCODE='22023';
    END IF;
    v_http_status := (p_review_doc ->> 'provider_http_status')::INTEGER;$new492a$);
  v_updated:=pg_catalog.replace(v_updated,$old492b$      'retryable', v_retryable
    ));$old492b$,
    $new492b$      'retryable', v_retryable,
      'provider_raw_http_status', p_review_doc->'provider_raw_http_status',
      'provider_effective_http_status', p_review_doc->'provider_effective_http_status',
      'provider_limit_source', p_review_doc->>'provider_limit_source'
    ));$new492b$);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena review492 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
      <>v_postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
        FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena review492 readback differs' USING ERRCODE='55000';
  END IF;
END;
$review_provider_observations492$;

COMMIT;
