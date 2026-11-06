-- Use the existing alternate route for only the final invalid-review slot.
-- Preserve the three-reservation cap, refusal episode routes, every deadline,
-- all source/ledger data, claim locks, costs, ACLs and completed outcomes.
-- Apply only through the canonical drained-release flow.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
LOCK TABLE public.lab_arena_rounds, public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
DO $alternate634$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)';
  v_preimage CONSTANT TEXT := '64b59849b4c36c19b3295fdb12cf8213171f8e550f903d79aaa3d463571cb079';
  v_postimage CONSTANT TEXT := '2bd23a14f1ba30bfdb4f4425c2adfef3226631301f41534143de5b49876065ee';
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
    RAISE EXCEPTION 'Arena alternate634 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena alternate634 preimage differs' USING ERRCODE='55000';
  END IF;
  v_old:=$old634$  ELSIF p_review_model='openai/gpt-6.1-sol' THEN
    RETURN pg_catalog.jsonb_build_object('status','fallback_not_authorized');$old634$;
  v_new:=$new634$  ELSIF NOT v_refusal_recovery AND v_invalid_start IS NOT NULL
        AND v_invalid_count>=2 AND v_invalid_count<3 THEN
    -- Spend only the last existing invalid-review reservation on the alternate.
    IF p_review_model IS DISTINCT FROM 'openai/gpt-6.1-sol' THEN
      RETURN pg_catalog.jsonb_build_object('status','route_required','review_model','openai/gpt-6.1-sol');
    END IF;
  ELSIF p_review_model='openai/gpt-6.1-sol' THEN
    RETURN pg_catalog.jsonb_build_object('status','fallback_not_authorized');$new634$;
  IF length(v_definition)-length(pg_catalog.replace(v_definition,v_old,''))<>length(v_old) THEN
    RAISE EXCEPTION 'Arena alternate634 anchor differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena alternate634 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>v_postimage
     OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid)
       IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena alternate634 readback differs' USING ERRCODE='55000';
  END IF;
END;
$alternate634$;
COMMIT;
