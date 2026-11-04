-- Exact November 3 admission recovery. Preserve original sources, judgments,
-- reservations and bills. This grants a bounded review, never admission.
-- Stop/drain review workers before apply. The normal 599 budget remains intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
LOCK TABLE public.lab_arena_rounds,public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
CREATE TEMP TABLE arena605_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $helper605$
DECLARE p RECORD;
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang
 WHERE proc.oid=to_regprocedure('public.lab_arena__nov03_review605_v1(public.lab_arena_rounds,public.lab_arena_submissions)');
 IF FOUND AND (p.prosrc IS DISTINCT FROM $expected605$
 SELECT coalesce(p_round.round_id='arena-2026-11-03' AND p_round.status='stage2'
  AND p_round.published_at IS NULL AND p_round.publication_doc IS NULL
  AND p_round.reward_activated_at IS NULL AND p_round.scoring_generation_floor=0
  AND p_round.configuration_doc#>>'{schedule,final_scoring_close}'='2026-11-03T20:30:02Z'
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND NOT p_submission.is_king
  AND p_submission.submission_id IN('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a')
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_round.round_id AND h.request_id='nov03-admission-review605-v1'
    AND h.recovery_doc->>'operation'='nov03_admission_review605_v1'
    AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc
    AND h.recovery_doc->'originals'->p_submission.submission_id->>'immutable_sha256'=
     encode(extensions.digest((to_jsonb(p_submission)-ARRAY['status','updated_at','code_review_status','code_review_doc','code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])::TEXT,'sha256'),'hex'))
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_round.round_id AND h.request_id='nov03-forward-only-verifier604-v1'
    AND h.recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
    AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc),FALSE);
$expected605$
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'sql') THEN
  RAISE EXCEPTION 'Nov03 admission605 helper differs'; END IF;
END;
$helper605$;
CREATE OR REPLACE FUNCTION public.lab_arena__nov03_review605_v1(
 p_round public.lab_arena_rounds,p_submission public.lab_arena_submissions)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $review605$
 SELECT coalesce(p_round.round_id='arena-2026-11-03' AND p_round.status='stage2'
  AND p_round.published_at IS NULL AND p_round.publication_doc IS NULL
  AND p_round.reward_activated_at IS NULL AND p_round.scoring_generation_floor=0
  AND p_round.configuration_doc#>>'{schedule,final_scoring_close}'='2026-11-03T20:30:02Z'
  AND p_submission.round_id=p_round.round_id AND p_submission.status='accepted'
  AND NOT p_submission.is_king
  AND p_submission.submission_id IN('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a')
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_round.round_id AND h.request_id='nov03-admission-review605-v1'
    AND h.recovery_doc->>'operation'='nov03_admission_review605_v1'
    AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc
    AND h.recovery_doc->'originals'->p_submission.submission_id->>'immutable_sha256'=
     encode(extensions.digest((to_jsonb(p_submission)-ARRAY['status','updated_at','code_review_status','code_review_doc','code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])::TEXT,'sha256'),'hex'))
  AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
   WHERE h.round_id=p_round.round_id AND h.request_id='nov03-forward-only-verifier604-v1'
    AND h.recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
    AND h.new_round_doc->'configuration_doc'=p_round.configuration_doc),FALSE);
$review605$;
ALTER FUNCTION public.lab_arena__nov03_review605_v1(public.lab_arena_rounds,public.lab_arena_submissions) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__nov03_review605_v1(public.lab_arena_rounds,public.lab_arena_submissions)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Read-only authority for the existing Store legacy-proof hydration. This
-- projects only an exact receipt-bound boolean; it cannot reserve or approve.
DO $reader605$
DECLARE p RECORD;
BEGIN
 SELECT proc.* INTO p FROM pg_proc proc WHERE proc.oid=to_regprocedure(
  'public.lab_arena_nov03_review_recovery_authorized_v1(text)');
 IF FOUND AND (p.prosrc IS DISTINCT FROM $reader_source605$
 SELECT EXISTS(SELECT 1 FROM public.lab_arena_rounds r
  JOIN public.lab_arena_submissions s ON s.round_id=r.round_id
  WHERE s.submission_id=p_submission_id
   AND public.lab_arena__nov03_review605_v1(r,s)
   AND clock_timestamp()<LEAST((r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ,
    (r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ));
$reader_source605$
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.prolang IS DISTINCT FROM (SELECT oid FROM pg_language WHERE lanname='sql')) THEN
  RAISE EXCEPTION 'Nov03 admission605 reader differs'; END IF;
END;
$reader605$;
CREATE OR REPLACE FUNCTION public.lab_arena_nov03_review_recovery_authorized_v1(p_submission_id TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $reader_source605$
 SELECT EXISTS(SELECT 1 FROM public.lab_arena_rounds r
  JOIN public.lab_arena_submissions s ON s.round_id=r.round_id
  WHERE s.submission_id=p_submission_id
   AND public.lab_arena__nov03_review605_v1(r,s)
   AND clock_timestamp()<LEAST((r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ,
    (r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ));
$reader_source605$;
ALTER FUNCTION public.lab_arena_nov03_review_recovery_authorized_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_nov03_review_recovery_authorized_v1(TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_nov03_review_recovery_authorized_v1(TEXT) TO lab_arena_service;

DO $patch605_0$
DECLARE d TEXT; updated TEXT; m JSONB; p RECORD; target OID:=to_regprocedure('public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)');
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang WHERE proc.oid=target;
 IF NOT FOUND OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.prosecdef IS DISTINCT FROM TRUE
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' OR p.provolatile<>'v'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql' THEN
  RAISE EXCEPTION 'Nov03 admission605 security differs: 0'; END IF;
 SELECT pg_get_functiondef(target),to_jsonb(proc)-'prosrc' INTO d,m FROM pg_proc proc WHERE proc.oid=target;
 IF encode(extensions.digest(d,'sha256'),'hex')='6c9974bae7c8b42dedcdee6e4a9145672402f5e0c74c87f8a9fff56afafeddf5' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'4b6560038d304138cccd174ed080313328c098a399bc8d50042099bc6d21986b' THEN
  RAISE EXCEPTION 'Nov03 admission605 preimage differs: 0'; END IF;
 updated:=d;
 IF length(updated)-length(replace(updated,$old605_0_0$  v_required_model TEXT;
  v_round_id TEXT;
$old605_0_0$,''))<>length($old605_0_0$  v_required_model TEXT;
  v_round_id TEXT;
$old605_0_0$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.0'; END IF;
 updated:=replace(updated,$old605_0_0$  v_required_model TEXT;
  v_round_id TEXT;
$old605_0_0$,$new605_0_0$  v_required_model TEXT;
  v_nov03_recovery BOOLEAN;
  v_round_id TEXT;
$new605_0_0$);
 IF length(updated)-length(replace(updated,$old605_0_1$  END IF;
  -- Follow the existing round/submission lock order before control SHARE.
$old605_0_1$,''))<>length($old605_0_1$  END IF;
  -- Follow the existing round/submission lock order before control SHARE.
$old605_0_1$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.1'; END IF;
 updated:=replace(updated,$old605_0_1$  END IF;
  -- Follow the existing round/submission lock order before control SHARE.
$old605_0_1$,$new605_0_1$  END IF;
  v_nov03_recovery := public.lab_arena__nov03_review605_v1(v_round,v_submission);
  -- Follow the existing round/submission lock order before control SHARE.
$new605_0_1$);
 IF length(updated)-length(replace(updated,$old605_0_2$
  v_is_baseline :=
$old605_0_2$,''))<>length($old605_0_2$
  v_is_baseline :=
$old605_0_2$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.2'; END IF;
 updated:=replace(updated,$old605_0_2$
  v_is_baseline :=
$old605_0_2$,$new605_0_2$
  IF v_nov03_recovery THEN
    v_deadline := LEAST((v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ, (v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ);
  END IF;
  v_is_baseline :=
$new605_0_2$);
 IF length(updated)-length(replace(updated,$old605_0_3$            WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
            WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
            ELSE 'code_review_transport_error'
$old605_0_3$,''))<>length($old605_0_3$            WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
            WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
            ELSE 'code_review_transport_error'
$old605_0_3$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.3'; END IF;
 updated:=replace(updated,$old605_0_3$            WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
            WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
            ELSE 'code_review_transport_error'
$old605_0_3$,$new605_0_3$            WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
            WHEN (v_round.status <> 'open' AND NOT v_nov03_recovery) THEN 'code_review_round_closed'
            ELSE 'code_review_transport_error'
$new605_0_3$);
 IF length(updated)-length(replace(updated,$old605_0_4$          END,
          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND (NOT v_refusal_recovery OR v_alternate_count<2),
$old605_0_4$,''))<>length($old605_0_4$          END,
          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND (NOT v_refusal_recovery OR v_alternate_count<2),
$old605_0_4$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.4'; END IF;
 updated:=replace(updated,$old605_0_4$          END,
          'retryable', v_now < v_deadline AND v_round.status = 'open'
            AND (NOT v_refusal_recovery OR v_alternate_count<2),
$old605_0_4$,$new605_0_4$          END,
          'retryable', v_now < v_deadline AND (v_round.status = 'open' OR v_nov03_recovery)
            AND (NOT v_refusal_recovery OR v_alternate_count<2),
$new605_0_4$);
 IF length(updated)-length(replace(updated,$old605_0_5$        WHEN v_now >= v_deadline THEN 'deadline'
        WHEN v_round.status <> 'open' THEN 'round_closed'
        ELSE 'recovered'
$old605_0_5$,''))<>length($old605_0_5$        WHEN v_now >= v_deadline THEN 'deadline'
        WHEN v_round.status <> 'open' THEN 'round_closed'
        ELSE 'recovered'
$old605_0_5$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.5'; END IF;
 updated:=replace(updated,$old605_0_5$        WHEN v_now >= v_deadline THEN 'deadline'
        WHEN v_round.status <> 'open' THEN 'round_closed'
        ELSE 'recovered'
$old605_0_5$,$new605_0_5$        WHEN v_now >= v_deadline THEN 'deadline'
        WHEN (v_round.status <> 'open' AND NOT v_nov03_recovery) THEN 'round_closed'
        ELSE 'recovered'
$new605_0_5$);
 IF length(updated)-length(replace(updated,$old605_0_6$
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
$old605_0_6$,''))<>length($old605_0_6$
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
$old605_0_6$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 0.6'; END IF;
 updated:=replace(updated,$old605_0_6$
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
$old605_0_6$,$new605_0_6$
  IF (v_round.status <> 'open' AND NOT v_nov03_recovery) THEN
    RETURN pg_catalog.jsonb_build_object(
$new605_0_6$);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'6c9974bae7c8b42dedcdee6e4a9145672402f5e0c74c87f8a9fff56afafeddf5' THEN
  RAISE EXCEPTION 'Nov03 admission605 postimage differs: 0'; END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'6c9974bae7c8b42dedcdee6e4a9145672402f5e0c74c87f8a9fff56afafeddf5' THEN
  RAISE EXCEPTION 'Nov03 admission605 readback differs: 0'; END IF;
END;
$patch605_0$;

DO $patch605_1$
DECLARE d TEXT; updated TEXT; m JSONB; p RECORD; target OID:=to_regprocedure('public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)');
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang WHERE proc.oid=target;
 IF NOT FOUND OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.prosecdef IS DISTINCT FROM TRUE
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' OR p.provolatile<>'v'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql' THEN
  RAISE EXCEPTION 'Nov03 admission605 security differs: 1'; END IF;
 SELECT pg_get_functiondef(target),to_jsonb(proc)-'prosrc' INTO d,m FROM pg_proc proc WHERE proc.oid=target;
 IF encode(extensions.digest(d,'sha256'),'hex')='947841e794d9be0b7c997d5752f892a26952d4fae81ddf01ecb60b7fac25de24' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'881ed3c299f5989d1e481416ec645ba2686ab646596a4aae4eb0f2aa2807e8a6' THEN
  RAISE EXCEPTION 'Nov03 admission605 preimage differs: 1'; END IF;
 updated:=d;
 IF length(updated)-length(replace(updated,$old605_1_0$  v_required_model TEXT;
  v_round_id TEXT;
$old605_1_0$,''))<>length($old605_1_0$  v_required_model TEXT;
  v_round_id TEXT;
$old605_1_0$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 1.0'; END IF;
 updated:=replace(updated,$old605_1_0$  v_required_model TEXT;
  v_round_id TEXT;
$old605_1_0$,$new605_1_0$  v_required_model TEXT;
  v_nov03_recovery BOOLEAN;
  v_round_id TEXT;
$new605_1_0$);
 IF length(updated)-length(replace(updated,$old605_1_1$  END IF;
  v_now := pg_catalog.clock_timestamp();
$old605_1_1$,''))<>length($old605_1_1$  END IF;
  v_now := pg_catalog.clock_timestamp();
$old605_1_1$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 1.1'; END IF;
 updated:=replace(updated,$old605_1_1$  END IF;
  v_now := pg_catalog.clock_timestamp();
$old605_1_1$,$new605_1_1$  END IF;
  v_nov03_recovery := public.lab_arena__nov03_review605_v1(v_round,v_submission);
  v_now := pg_catalog.clock_timestamp();
$new605_1_1$);
 IF length(updated)-length(replace(updated,$old605_1_2$  );
  v_is_baseline :=
$old605_1_2$,''))<>length($old605_1_2$  );
  v_is_baseline :=
$old605_1_2$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 1.2'; END IF;
 updated:=replace(updated,$old605_1_2$  );
  v_is_baseline :=
$old605_1_2$,$new605_1_2$  );
  IF v_nov03_recovery THEN
    v_deadline := LEAST((v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ, (v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ);
  END IF;
  v_is_baseline :=
$new605_1_2$);
 IF length(updated)-length(replace(updated,$old605_1_3$  v_now := pg_catalog.clock_timestamp();
  IF (v_now >= v_deadline OR v_round.status <> 'open'
      OR v_submission.status <> 'accepted')
$old605_1_3$,''))<>length($old605_1_3$  v_now := pg_catalog.clock_timestamp();
  IF (v_now >= v_deadline OR v_round.status <> 'open'
      OR v_submission.status <> 'accepted')
$old605_1_3$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 1.3'; END IF;
 updated:=replace(updated,$old605_1_3$  v_now := pg_catalog.clock_timestamp();
  IF (v_now >= v_deadline OR v_round.status <> 'open'
      OR v_submission.status <> 'accepted')
$old605_1_3$,$new605_1_3$  v_now := pg_catalog.clock_timestamp();
  IF (v_now >= v_deadline OR (v_round.status <> 'open' AND NOT v_nov03_recovery)
      OR v_submission.status <> 'accepted')
$new605_1_3$);
 IF length(updated)-length(replace(updated,$old605_1_4$        WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
        WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
        ELSE 'code_review_submission_ineligible'
$old605_1_4$,''))<>length($old605_1_4$        WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
        WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
        ELSE 'code_review_submission_ineligible'
$old605_1_4$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 1.4'; END IF;
 updated:=replace(updated,$old605_1_4$        WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
        WHEN v_round.status <> 'open' THEN 'code_review_round_closed'
        ELSE 'code_review_submission_ineligible'
$old605_1_4$,$new605_1_4$        WHEN v_now >= v_deadline THEN 'code_review_deadline_exceeded'
        WHEN (v_round.status <> 'open' AND NOT v_nov03_recovery) THEN 'code_review_round_closed'
        ELSE 'code_review_submission_ineligible'
$new605_1_4$);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'947841e794d9be0b7c997d5752f892a26952d4fae81ddf01ecb60b7fac25de24' THEN
  RAISE EXCEPTION 'Nov03 admission605 postimage differs: 1'; END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'947841e794d9be0b7c997d5752f892a26952d4fae81ddf01ecb60b7fac25de24' THEN
  RAISE EXCEPTION 'Nov03 admission605 readback differs: 1'; END IF;
END;
$patch605_1$;

DO $patch605_2$
DECLARE d TEXT; updated TEXT; m JSONB; p RECORD; target OID:=to_regprocedure('public.lab_arena_submissions_frozen_v1()');
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang WHERE proc.oid=target;
 IF NOT FOUND OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR p.prosecdef IS DISTINCT FROM FALSE
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}' OR p.provolatile<>'v'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog']::TEXT[] OR p.lanname<>'plpgsql' THEN
  RAISE EXCEPTION 'Nov03 admission605 security differs: 2'; END IF;
 SELECT pg_get_functiondef(target),to_jsonb(proc)-'prosrc' INTO d,m FROM pg_proc proc WHERE proc.oid=target;
 IF encode(extensions.digest(d,'sha256'),'hex')='73a0ebc54d35f76b18fc0b1a83b3ab36ed9f793748274b3ea32ea96584da6fb1' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'3a1a312aa409c7e1d366255f4c56f64a022d2c755a89ab1ace0cfea5b6821d10' THEN
  RAISE EXCEPTION 'Nov03 admission605 preimage differs: 2'; END IF;
 updated:=d;
 IF length(updated)-length(replace(updated,$old605_2_0$  END IF;
  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' THEN
    RAISE EXCEPTION 'rejected submission cannot be reopened'
$old605_2_0$,''))<>length($old605_2_0$  END IF;
  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' THEN
    RAISE EXCEPTION 'rejected submission cannot be reopened'
$old605_2_0$) THEN
  RAISE EXCEPTION 'Nov03 admission605 anchor differs: 2.0'; END IF;
 updated:=replace(updated,$old605_2_0$  END IF;
  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' THEN
    RAISE EXCEPTION 'rejected submission cannot be reopened'
$old605_2_0$,$new605_2_0$  END IF;
  IF OLD.status = 'rejected' AND NEW.status <> 'rejected' AND NOT (
    NEW.status='accepted'
    AND (to_jsonb(NEW)-ARRAY['status','updated_at'])=(to_jsonb(OLD)-ARRAY['status','updated_at'])
    AND OLD.round_id='arena-2026-11-03'
    AND OLD.submission_id IN('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a')
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=OLD.round_id AND h.request_id='nov03-admission-review605-v1'
       AND h.recovery_doc->>'operation'='nov03_admission_review605_v1'
       AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->'originals'->OLD.submission_id->>'row_sha256'=
        encode(extensions.digest(to_jsonb(OLD)::TEXT,'sha256'),'hex'))
  ) THEN
    RAISE EXCEPTION 'rejected submission cannot be reopened'
$new605_2_0$);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'73a0ebc54d35f76b18fc0b1a83b3ab36ed9f793748274b3ea32ea96584da6fb1' THEN
  RAISE EXCEPTION 'Nov03 admission605 postimage differs: 2'; END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'73a0ebc54d35f76b18fc0b1a83b3ab36ed9f793748274b3ea32ea96584da6fb1' THEN
  RAISE EXCEPTION 'Nov03 admission605 readback differs: 2'; END IF;
END;
$patch605_2$;


-- Serialize the exact recovery window with normal stage closure. No new pause,
-- run mutation or deadline extension is introduced by this narrow predicate.
DO $close605$
DECLARE d TEXT; updated TEXT; m JSONB; p RECORD;
 target OID:=to_regprocedure('public.lab_arena_close_stage(text,smallint)');
BEGIN
 SELECT proc.*,l.lanname INTO p FROM pg_proc proc JOIN pg_language l ON l.oid=proc.prolang WHERE proc.oid=target;
 IF NOT FOUND OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  OR p.provolatile<>'v' OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[]
  OR p.lanname<>'plpgsql' THEN RAISE EXCEPTION 'Nov03 admission605 close security differs'; END IF;
 SELECT pg_get_functiondef(target),to_jsonb(proc)-'prosrc' INTO d,m FROM pg_proc proc WHERE proc.oid=target;
 IF encode(extensions.digest(d,'sha256'),'hex')='96db06408278ccad9cca9566457518c2951c944d33149c8efe2336c8cca164aa' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dc50c521f73ff996c28437ce3321d62cecbf4a5adfb69126f94a35708ecde481'
  OR length(d)-length(replace(d,$close_anchor605$  -- lab_arena_stage_cutoff_drain_v1: stop admission at the frozen cutoff,$close_anchor605$,''))<>length($close_anchor605$  -- lab_arena_stage_cutoff_drain_v1: stop admission at the frozen cutoff,$close_anchor605$) THEN
  RAISE EXCEPTION 'Nov03 admission605 close preimage differs'; END IF;
 updated:=replace(d,$close_anchor605$  -- lab_arena_stage_cutoff_drain_v1: stop admission at the frozen cutoff,$close_anchor605$,$close_new605$  -- The two authorized originals have no runs until their bounded reviews
  -- and owner finalization finish. Existing runs alone cannot close their
  -- admission window early. Execution and judging continue at the same clock.
  IF p_stage=2 AND v_round.round_id='arena-2026-11-03'
    AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ
    AND NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      WHERE h.round_id=v_round.round_id AND h.request_id='nov03-admit-reviewed-originals606-v1')
    AND (SELECT count(*)=2 FROM public.lab_arena_submissions s
      WHERE s.submission_id IN('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a')
       AND public.lab_arena__nov03_review605_v1(v_round,s)) THEN
    RETURN jsonb_build_object('status','waiting','round_status',v_round.status,
      'stage_generation',v_round.stage_generation,'reason','admission_recovery_pending');
  END IF;

  -- lab_arena_stage_cutoff_drain_v1: stop admission at the frozen cutoff,$close_new605$);
 IF encode(extensions.digest(updated,'sha256'),'hex')<>'96db06408278ccad9cca9566457518c2951c944d33149c8efe2336c8cca164aa' THEN
  RAISE EXCEPTION 'Nov03 admission605 close postimage differs'; END IF;
 EXECUTE updated;
 IF (SELECT to_jsonb(proc)-'prosrc' FROM pg_proc proc WHERE proc.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'96db06408278ccad9cca9566457518c2951c944d33149c8efe2336c8cca164aa' THEN
  RAISE EXCEPTION 'Nov03 admission605 close readback differs'; END IF;
END;
$close605$;

DO $recover605$
DECLARE r public.lab_arena_rounds; s public.lab_arena_submissions;
 h public.lab_arena_scoring_supersessions; c public.lab_arena_restart_claim_control;
 item RECORD; originals JSONB:='{}'; identity TEXT; reservation public.lab_arena_ledger;
 head public.lab_arena_ledger; generation BIGINT;
BEGIN
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id='arena-2026-11-03' FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions
 WHERE round_id=r.round_id AND request_id='nov03-admission-review605-v1';
 IF FOUND THEN
  IF h.recovery_doc->>'operation' IS DISTINCT FROM 'nov03_admission_review605_v1'
   OR h.new_round_doc->'configuration_doc' IS DISTINCT FROM r.configuration_doc THEN
   RAISE EXCEPTION 'Nov03 admission605 replay differs'; END IF;
  RETURN; -- Never reset a review, spent claim or completed decision.
 END IF;
 SELECT * INTO STRICT c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 IF current_setting('session_replication_role')<>'origin'
  OR NOT c.operator_paused OR c.pause_reason<>'nov03_improved_verifier_required603'
  OR r.status<>'stage2' OR r.scoring_generation_floor<>0
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL OR r.reward_activated_at IS NOT NULL
  OR r.configuration_doc->>'execution_sequence_policy'<>'baseline_scored_first_v1'
  OR r.configuration_doc#>>'{schedule,final_scoring_close}'<>'2026-11-03T20:30:02Z'
  OR clock_timestamp()>=LEAST((r.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ,
    (r.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ)
  OR NOT EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions verifier
   WHERE verifier.round_id=r.round_id AND verifier.request_id='nov03-forward-only-verifier604-v1'
    AND verifier.recovery_doc->>'operation'='nov03_forward_only_verifier_v1'
    AND verifier.new_round_doc->'configuration_doc'=r.configuration_doc)
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE status IN('leased','submitted'))
  OR EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE code_review_status='reviewing') THEN
  RAISE EXCEPTION 'Nov03 admission605 round, hold, drain or verifier differs'; END IF;
 FOR item IN SELECT * FROM (VALUES
('sub-28594d902080fb5213f4344c09ffb531','b0b27246c1b953694b1559737f1e78a9dc31dc4f3d55ff7d64b89567a00a64ae','sha256:cd4c88c88f38d6bc4140b3f64ca088fad246663f63d96555909fe9f589c6a894'),
('sub-ba551886a9379aa4e00b6238567f211a','7a6ff726c4644dd3c28d94e2293967898c4d7dc9d6d9a86d20369463dc5a3b74','sha256:fc270d967ac39c835d9ccb2e5d909289911c9b38e3b919cf68149e18f032d967')) AS targets(submission_id,row_sha256,source_sha256) LOOP
  SELECT * INTO STRICT s FROM public.lab_arena_submissions WHERE submission_id=item.submission_id FOR UPDATE;
  IF encode(extensions.digest(to_jsonb(s)::TEXT,'sha256'),'hex')<>item.row_sha256
   OR s.round_id<>r.round_id OR s.status<>'rejected' OR s.is_king
   OR s.source_archive_sha256<>item.source_sha256
   OR s.code_review_status<>'error' OR s.code_review_attempts<>1
   OR s.code_review_doc->>'error_code'<>'code_review_provider_request_rejected'
   OR s.code_review_doc->'provider_http_status'<>'200'::JSONB
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE submission_id=s.submission_id)
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(r.participants)p WHERE p->>'submission_id'=s.submission_id) THEN
   RAISE EXCEPTION 'Nov03 admission605 original differs: %',item.submission_id; END IF;
  identity:='sha256:'||encode(extensions.digest('lab_arena.code_review:'||s.submission_id||':1','sha256'),'hex');
  SELECT * INTO STRICT reservation FROM public.lab_arena_ledger WHERE call_identity=identity AND entry_kind='reservation';
  head:=public.lab_arena__ledger_head(identity);
  IF reservation.submission_id<>s.submission_id OR reservation.round_id<>r.round_id
   OR reservation.operation_id<>'openrouter.code_review' OR reservation.entry_doc->'attempt'<>'1'::JSONB
   OR reservation.entry_doc->>'review_model'<>'anthropic/claude-sonnet-5.5'
   OR head.entry_kind NOT IN('settlement','uncertain') OR head.submission_id<>s.submission_id
   OR head.entry_doc->>'error_code'<>'code_review_provider_request_rejected'
   OR head.entry_doc->'provider_http_status'<>'200'::JSONB
   OR (SELECT count(*) FROM public.lab_arena_ledger WHERE submission_id=s.submission_id AND operation_id='openrouter.code_review')<>3 THEN
   RAISE EXCEPTION 'Nov03 admission605 authoritative refusal differs'; END IF;
  originals:=originals||jsonb_build_object(s.submission_id,jsonb_build_object(
   'row_sha256',item.row_sha256,'source_archive_sha256',s.source_archive_sha256,
   'immutable_sha256',encode(extensions.digest((to_jsonb(s)-ARRAY['status','updated_at','code_review_status','code_review_doc','code_review_claim','code_review_started_at','code_review_expires_at','code_review_attempts']::TEXT[])::TEXT,'sha256'),'hex'),
   'original_code_review_doc',s.code_review_doc,'original_rejection_rule',s.rejection_rule,
   'original_status',s.status,'original_attempts',s.code_review_attempts,
   'reservation',to_jsonb(reservation),'terminal',to_jsonb(head)));
 END LOOP;
 SELECT coalesce(max(generation_floor),0)+1 INTO generation FROM public.lab_arena_scoring_supersessions WHERE round_id=r.round_id;
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,
  actor_ref,transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(r.round_id,generation,'nov03-admission-review605-v1',
  'sha256:'||encode(extensions.digest(originals::TEXT,'sha256'),'hex'),'operator:nov03-admission605',
  txid_current(),pg_backend_pid(),to_jsonb(r),to_jsonb(r),'{}',
  jsonb_build_object('operation','nov03_admission_review605_v1','originals',originals,
   'scoring_generation_floor',r.scoring_generation_floor,'primary_claims_total',2,'alternate_claims_total',2));
 UPDATE public.lab_arena_submissions SET status='accepted'
 WHERE submission_id IN('sub-28594d902080fb5213f4344c09ffb531','sub-ba551886a9379aa4e00b6238567f211a');
 IF (SELECT count(*) FROM public.lab_arena_submissions WHERE round_id=r.round_id
  AND public.lab_arena__nov03_review605_v1(r,lab_arena_submissions))<>2 THEN
  RAISE EXCEPTION 'Nov03 admission605 readback differs'; END IF;
END;
$recover605$;
DO $restore605$
BEGIN
 IF NOT (SELECT had_create FROM arena605_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM arena605_schema_acl) THEN
  RAISE EXCEPTION 'Nov03 admission605 schema ACL differs'; END IF;
END;
$restore605$;
NOTIFY pgrst,'reload schema';
COMMIT;
