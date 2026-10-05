-- Read the submission-wide cost document once for each published ranking.
-- The private JSON summarizer retains the exact aggregation from migration 289.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $publication_cost_409$
DECLARE
  v_wrapper TEXT;
  v_guard TEXT;
  v_helper TEXT;
  v_updated_wrapper TEXT;
  v_updated_guard TEXT;
  v_wrapper_identity JSONB;
  v_guard_identity JSONB;
  v_helper_identity JSONB;
  v_old_calls TEXT := $old$  v_expected_execution := public.lab_arena__cost_kind_summary_v1(
    v_submission_id, 'execute'
  );
  v_expected_judge := public.lab_arena__cost_kind_summary_v1(
    v_submission_id, 'score'
  );$old$;
  v_new_calls TEXT := $new$  v_cost_document := public.lab_arena_submission_costs(v_submission_id);
  v_expected_execution := public.lab_arena__cost_kind_summary_from_doc_v1(
    v_cost_document, 'execute'
  );
  v_expected_judge := public.lab_arena__cost_kind_summary_from_doc_v1(
    v_cost_document, 'score'
  );$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid),
         pg_catalog.jsonb_build_array(owner.rolname, procedure.proacl::TEXT,
           procedure.prosecdef, procedure.provolatile, procedure.proconfig)
  INTO v_wrapper, v_wrapper_identity
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena__cost_kind_summary_v1'
    AND procedure.pronargs = 2;
  SELECT pg_catalog.pg_get_functiondef(procedure.oid),
         pg_catalog.jsonb_build_array(owner.rolname, procedure.proacl::TEXT,
           procedure.prosecdef, procedure.provolatile, procedure.proconfig)
  INTO v_guard, v_guard_identity
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena__per_icp_publication_valid'
    AND procedure.pronargs = 2;
  SELECT pg_catalog.jsonb_build_array(owner.rolname, procedure.proacl::TEXT,
           procedure.prosecdef, procedure.provolatile, procedure.proconfig)
  INTO v_helper_identity
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE procedure.oid = pg_catalog.to_regprocedure(
    'public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)'
  );
  IF v_wrapper IS NULL OR v_guard IS NULL
     OR v_wrapper_identity ->> 0 IS DISTINCT FROM 'lab_arena_owner'
     OR v_guard_identity ->> 0 IS DISTINCT FROM 'lab_arena_owner'
     OR v_wrapper_identity ->> 2 IS DISTINCT FROM 'true'
     OR v_guard_identity ->> 2 IS DISTINCT FROM 'true'
     OR v_wrapper_identity ->> 3 IS DISTINCT FROM 's'
     OR v_guard_identity ->> 3 IS DISTINCT FROM 's' THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_409_security_shape_changed';
  END IF;
  -- These hashes are of the exact live definitions immediately after 408.
  IF pg_catalog.md5(v_wrapper) = '2b04ca837f37e445c5e71d31d30d4b8f'
     AND pg_catalog.md5(v_guard) = '478a73576821d3bf34fe9c2176f3d46f'
     AND pg_catalog.md5(pg_catalog.pg_get_functiondef(
       pg_catalog.to_regprocedure(
         'public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)'
       )
     )) = '2777c8ed895fcbd3730e6673959e6767'
     AND v_helper_identity IS NOT DISTINCT FROM v_wrapper_identity THEN
    RETURN;
  END IF;
  IF pg_catalog.md5(v_wrapper) <> '6e5b992a26e2639f9894cbbff2bbe4cb'
     OR pg_catalog.md5(v_guard) <> '67b6e1ccddb42548262d423a048a2f5e'
     OR pg_catalog.to_regprocedure(
       'public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)'
     ) IS NOT NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_409_preimage_changed';
  END IF;

  v_helper := pg_catalog.replace(
    v_wrapper,
    'public.lab_arena__cost_kind_summary_v1(p_submission_id text, p_kind text)',
    'public.lab_arena__cost_kind_summary_from_doc_v1(p_costs jsonb, p_kind text)'
  );
  v_helper := pg_catalog.replace(
    v_helper, 'public.lab_arena_submission_costs(p_submission_id)', 'p_costs'
  );
  IF pg_catalog.md5(v_helper) <> '2777c8ed895fcbd3730e6673959e6767' THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_409_helper_invalid';
  END IF;
  EXECUTE v_helper;
  ALTER FUNCTION public.lab_arena__cost_kind_summary_from_doc_v1(JSONB,TEXT)
    OWNER TO lab_arena_owner;
  REVOKE ALL ON FUNCTION public.lab_arena__cost_kind_summary_from_doc_v1(JSONB,TEXT)
    FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

  v_updated_wrapper := pg_catalog.split_part(v_wrapper, 'AS $function$', 1)
    || $wrapper$AS $function$
  SELECT public.lab_arena__cost_kind_summary_from_doc_v1(
    public.lab_arena_submission_costs(p_submission_id), p_kind
  );
$function$
$wrapper$;
  v_updated_guard := pg_catalog.replace(
    pg_catalog.replace(
      v_guard, '  v_expected_execution JSONB;',
      E'  v_cost_document JSONB;\n  v_expected_execution JSONB;'
    ), v_old_calls, v_new_calls
  );
  IF pg_catalog.md5(v_updated_wrapper) <> '2b04ca837f37e445c5e71d31d30d4b8f'
     OR pg_catalog.md5(v_updated_guard) <> '478a73576821d3bf34fe9c2176f3d46f' THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_409_postimage_invalid';
  END IF;
  EXECUTE v_updated_wrapper;
  EXECUTE v_updated_guard;

  IF pg_catalog.md5(pg_catalog.pg_get_functiondef(
       'public.lab_arena__cost_kind_summary_v1(text,text)'::REGPROCEDURE
     )) <> '2b04ca837f37e445c5e71d31d30d4b8f'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
       'public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)'::REGPROCEDURE
     )) <> '2777c8ed895fcbd3730e6673959e6767'
     OR pg_catalog.md5(pg_catalog.pg_get_functiondef(
       'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE
     )) <> '478a73576821d3bf34fe9c2176f3d46f'
     OR (SELECT pg_catalog.jsonb_build_array(owner.rolname, procedure.proacl::TEXT,
           procedure.prosecdef, procedure.provolatile, procedure.proconfig)
         FROM pg_catalog.pg_proc AS procedure
         JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
         WHERE procedure.oid =
           'public.lab_arena__cost_kind_summary_v1(text,text)'::REGPROCEDURE)
       IS DISTINCT FROM v_wrapper_identity
     OR (SELECT pg_catalog.jsonb_build_array(owner.rolname, procedure.proacl::TEXT,
           procedure.prosecdef, procedure.provolatile, procedure.proconfig)
         FROM pg_catalog.pg_proc AS procedure
         JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
         WHERE procedure.oid =
           'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE)
       IS DISTINCT FROM v_guard_identity
     OR (SELECT pg_catalog.jsonb_build_array(owner.rolname, procedure.proacl::TEXT,
           procedure.prosecdef, procedure.provolatile, procedure.proconfig)
         FROM pg_catalog.pg_proc AS procedure
         JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
         WHERE procedure.oid =
           'public.lab_arena__cost_kind_summary_from_doc_v1(jsonb,text)'::REGPROCEDURE)
       IS DISTINCT FROM v_wrapper_identity THEN
    RAISE EXCEPTION 'lab_arena_publication_cost_409_readback_changed';
  END IF;
END;
$publication_cost_409$;

NOTIFY pgrst, 'reload schema';
COMMIT;
