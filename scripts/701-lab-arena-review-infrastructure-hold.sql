-- Hold an open round, instead of rejecting its miners, while our reviewer
-- infrastructure (provider route, timeout, transport) blocks their reviews.
-- An unreleased hold extends only the SQL review window of that open round,
-- bounded at stage_1_start + 120 minutes. Dated recoveries, reservations,
-- ledger rules, verdicts, stored rows and published rounds stay unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
CREATE TEMP TABLE arena701_acl ON COMMIT DROP AS SELECT nspacl,
 has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_review_holds (
  round_id TEXT PRIMARY KEY REFERENCES public.lab_arena_rounds(round_id),
  review_deadline TIMESTAMPTZ NOT NULL,
  max_review_deadline TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  extended_at TIMESTAMPTZ,
  released_at TIMESTAMPTZ,
  CONSTRAINT lab_arena_review_holds_bounded CHECK (review_deadline <= max_review_deadline)
);
ALTER TABLE public.lab_arena_review_holds OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_review_holds ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.lab_arena_review_holds FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- The review window an unreleased hold grants an open round, else NULL.
CREATE OR REPLACE FUNCTION public.lab_arena__review_hold_deadline701_v1(p_round public.lab_arena_rounds)
RETURNS TIMESTAMPTZ LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $deadline701$
  SELECT h.review_deadline FROM public.lab_arena_review_holds h
  WHERE h.round_id=p_round.round_id AND h.released_at IS NULL AND p_round.status='open';
$deadline701$;
ALTER FUNCTION public.lab_arena__review_hold_deadline701_v1(public.lab_arena_rounds) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__review_hold_deadline701_v1(public.lab_arena_rounds)
  FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Idempotent: the first call after the submission cutoff fixes the window.
CREATE OR REPLACE FUNCTION public.lab_arena_hold_round_review_v1(p_round_id TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $hold701$
DECLARE
  v_round public.lab_arena_rounds;
  v_hold public.lab_arena_review_holds;
  v_stage_1 TIMESTAMPTZ;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE='P0002';
  END IF;
  SELECT * INTO v_hold FROM public.lab_arena_review_holds WHERE round_id=p_round_id;
  IF FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status','existing','review_deadline',v_hold.review_deadline,
      'max_review_deadline',v_hold.max_review_deadline,'released_at',v_hold.released_at);
  END IF;
  IF v_round.status<>'open' THEN
    RETURN pg_catalog.jsonb_build_object('status','not_open');
  END IF;
  IF pg_catalog.clock_timestamp()<(v_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ THEN
    RETURN pg_catalog.jsonb_build_object('status','not_due');
  END IF;
  v_stage_1:=COALESCE(
    (v_round.configuration_doc#>>'{schedule,stage_1_start}')::TIMESTAMPTZ,
    (v_round.configuration_doc#>>'{schedule,benchmark_deadline}')::TIMESTAMPTZ+INTERVAL '1 second',
    (v_round.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ+INTERVAL '30 minutes 1 second');
  INSERT INTO public.lab_arena_review_holds(round_id,review_deadline,max_review_deadline)
  VALUES (p_round_id,v_stage_1+INTERVAL '60 minutes',v_stage_1+INTERVAL '120 minutes')
  RETURNING * INTO v_hold;
  RETURN pg_catalog.jsonb_build_object('status','held','review_deadline',v_hold.review_deadline,
    'max_review_deadline',v_hold.max_review_deadline,'released_at',v_hold.released_at);
END;
$hold701$;
ALTER FUNCTION public.lab_arena_hold_round_review_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_hold_round_review_v1(TEXT) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_hold_round_review_v1(TEXT) TO lab_arena_service;

-- Operator decision: extend once to the fixed cap, or release the round so
-- still-blocked entries take their pre-hold outcome. Both are one-way.
CREATE OR REPLACE FUNCTION public.lab_arena_decide_review_hold_v1(p_round_id TEXT,p_action TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $decide701$
DECLARE
  v_round public.lab_arena_rounds;
  v_hold public.lab_arena_review_holds;
BEGIN
  IF p_action IS NULL OR p_action NOT IN ('extend','release') THEN
    RAISE EXCEPTION 'lab_arena_review_hold_action_invalid' USING ERRCODE='22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE='P0002';
  END IF;
  SELECT * INTO v_hold FROM public.lab_arena_review_holds WHERE round_id=p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status','no_hold');
  END IF;
  IF v_hold.released_at IS NOT NULL THEN
    RETURN pg_catalog.jsonb_build_object('status','released','review_deadline',v_hold.review_deadline,
      'max_review_deadline',v_hold.max_review_deadline,'released_at',v_hold.released_at);
  END IF;
  IF p_action='release' THEN
    UPDATE public.lab_arena_review_holds SET released_at=pg_catalog.clock_timestamp()
    WHERE round_id=p_round_id RETURNING * INTO v_hold;
  ELSIF v_round.status<>'open' THEN
    RETURN pg_catalog.jsonb_build_object('status','not_open');
  ELSIF v_hold.review_deadline<v_hold.max_review_deadline THEN
    UPDATE public.lab_arena_review_holds
    SET review_deadline=max_review_deadline,extended_at=pg_catalog.clock_timestamp()
    WHERE round_id=p_round_id RETURNING * INTO v_hold;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status',CASE WHEN p_action='release' THEN 'released' ELSE 'extended' END,
    'review_deadline',v_hold.review_deadline,'max_review_deadline',v_hold.max_review_deadline,
    'released_at',v_hold.released_at);
END;
$decide701$;
ALTER FUNCTION public.lab_arena_decide_review_hold_v1(TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_decide_review_hold_v1(TEXT,TEXT) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_decide_review_hold_v1(TEXT,TEXT) TO lab_arena_service;

-- Anchored rewrites of the exact deployed review gate, finish and readiness
-- probe. Each is skipped when already at its post-image.
DO $patch701$
DECLARE
  target RECORD;
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_metadata JSONB;
  v_identity JSONB;
  v_anchor TEXT;
  v_index INTEGER;
  v_acl CONSTANT TEXT := '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}';
  v_hold_decl CONSTANT TEXT := $d701$  v_nov06_deadline TIMESTAMPTZ;$d701$;
  v_hold_decl_new CONSTANT TEXT := $d701$  v_nov06_deadline TIMESTAMPTZ;
  v_review_hold_deadline TIMESTAMPTZ;$d701$;
  v_hold_apply CONSTANT TEXT := $a701$  IF v_nov06_deadline IS NOT NULL THEN v_deadline:=v_nov06_deadline; END IF;$a701$;
  v_hold_apply_new CONSTANT TEXT := $a701$  IF v_nov06_deadline IS NOT NULL THEN v_deadline:=v_nov06_deadline; END IF;
  -- lab_arena_review_infrastructure_hold701: an unreleased hold on an open
  -- round extends its review window; dated recoveries keep their deadline.
  IF NOT COALESCE(v_nov03_recovery,FALSE) AND v_nov06_deadline IS NULL THEN
    v_review_hold_deadline:=public.lab_arena__review_hold_deadline701_v1(v_round);
    IF v_review_hold_deadline>v_deadline THEN v_deadline:=v_review_hold_deadline; END IF;
  END IF;$a701$;
  -- While held, a claim cut off by the window is our failure: record a
  -- retryable transport error, never the terminal deadline outcome.
  v_late CONSTANT TEXT := $l701$WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'$l701$;
  v_late_new CONSTANT TEXT := $l701$WHEN v_now >= v_deadline AND v_review_hold_deadline IS NOT NULL THEN 'code_review_transport_error'
        WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'$l701$;
BEGIN
  FOR target IN SELECT * FROM (VALUES
    ('public.lab_arena_begin_submission_review_v2(text,text,text,bigint,text,integer,bigint,text)','v',
     '00b9b8c403a5f446eef02e1839eec7d6f96c681bf55c5712b6bb6a2c05f8cbc5','d5bf3a7c86e97dfb1b7080f5d6e959b05c0fa3463a10ecf88f93e67fddbaba5a'),
    ('public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)','v',
     'a88ac51cd235480d73630e9a426004a58bdbac59699c88c2de31accc83c6a584','0c195df953c91fab3d7a1267d66b80cfc1d10a419cf349641915afd3b88504d9'),
    ('public.lab_arena_submission_replacement_schema_v1()','s',
     '078a88402b352debb9a3a03ad8e07df908da693b662f6fe9f542883e06ff8081','e39f8dd8162715b476dd02f7a6535512500fa0d29e8ccf9f43b2903361d3adbc')
  ) AS t(signature,volatility,preimage,postimage) LOOP
    v_oid:=pg_catalog.to_regprocedure(target.signature);
    SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.to_jsonb(p)-'prosrc',
      pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_metadata,v_identity FROM pg_catalog.pg_proc p WHERE p.oid=v_oid;
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
        'lab_arena_owner',v_acl,TRUE,target.volatility,ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'Arena hold701 security differs: %',target.signature USING ERRCODE='55000';
    END IF;
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=target.postimage THEN CONTINUE; END IF;
    IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex') IS DISTINCT FROM target.preimage THEN
      RAISE EXCEPTION 'Arena hold701 preimage differs: %',target.signature USING ERRCODE='55000';
    END IF;
    v_updated:=v_definition;
    IF target.volatility='v' THEN
      FOR v_index IN 1..3 LOOP
        v_anchor:=(ARRAY[v_hold_decl,v_hold_apply,v_late])[v_index];
        IF length(v_updated)-length(pg_catalog.replace(v_updated,v_anchor,''))<>length(v_anchor) THEN
          RAISE EXCEPTION 'Arena hold701 anchor differs: % %',target.signature,v_index USING ERRCODE='55000';
        END IF;
        v_updated:=pg_catalog.replace(v_updated,v_anchor,(ARRAY[v_hold_decl_new,v_hold_apply_new,v_late_new])[v_index]);
      END LOOP;
    ELSE
      -- The readiness probe pins the exact gate body it accepts.
      v_anchor:='''00b9b8c403a5f446eef02e1839eec7d6f96c681bf55c5712b6bb6a2c05f8cbc5''';
      IF length(v_updated)-length(pg_catalog.replace(v_updated,v_anchor,''))<>length(v_anchor) THEN
        RAISE EXCEPTION 'Arena hold701 anchor differs: %',target.signature USING ERRCODE='55000';
      END IF;
      v_updated:=pg_catalog.replace(v_updated,v_anchor,'''d5bf3a7c86e97dfb1b7080f5d6e959b05c0fa3463a10ecf88f93e67fddbaba5a''');
    END IF;
    IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>target.postimage THEN
      RAISE EXCEPTION 'Arena hold701 postimage differs: % %',target.signature,
        pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex') USING ERRCODE='55000';
    END IF;
    EXECUTE v_updated;
    IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')<>target.postimage
       OR (SELECT pg_catalog.to_jsonb(p)-'prosrc' FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
      RAISE EXCEPTION 'Arena hold701 readback differs: %',target.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$patch701$;

DO $restore701$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.arena701_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM(SELECT nspacl FROM pg_temp.arena701_acl) THEN
  RAISE EXCEPTION 'Arena hold701 schema ACL differs'; END IF;
END;
$restore701$;
NOTIFY pgrst,'reload schema';
COMMIT;
