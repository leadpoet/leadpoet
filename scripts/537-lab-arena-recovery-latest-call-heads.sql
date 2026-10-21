-- Preserve recovery drain authority while avoiding historical CID anti-probes.
-- The latest-head predicate is the one already used by cancellation533.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $recovery_call_heads537$
DECLARE target OID:='public.lab_arena_recover_evaluation_v1(text,jsonb,bigint,bigint,jsonb,jsonb,jsonb,text,text)'::regprocedure; d TEXT; metadata JSONB;
BEGIN
 IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
    WHERE i.indexrelid=to_regclass('public.lab_arena_ledger_round_call_head_idx')
      AND i.indrelid='public.lab_arena_ledger'::regclass AND i.indisvalid AND i.indisready AND i.indislive
      AND pg_get_userbyid(c.relowner)='lab_arena_owner'
      AND pg_get_indexdef(i.indexrelid)='CREATE INDEX lab_arena_ledger_round_call_head_idx ON public.lab_arena_ledger USING btree (round_id) WHERE (call_identity IS NOT NULL)') THEN
  RAISE EXCEPTION 'recovery round call head index differs' USING ERRCODE='55000';
 END IF;
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}' AND p.prosecdef
    AND p.provolatile='v' AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND l.lanname='plpgsql';
 IF encode(extensions.digest(d,'sha256'),'hex')='b66646eeb66663214623d5771feb60a03f6ce9821541341d2a68bb2423c02b84' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'5e4e3a89edd1c4578a0168d238d97acf9b4f62900a2368a992800f4ba4b75644'
   OR (length(d)-length(replace(d,$old537$  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.round_id=p_round_id AND l.call_identity IS NOT NULL
    AND l.entry_kind IN('reservation','dispatch') AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
      WHERE later.call_identity=l.call_identity AND later.entry_id>l.entry_id))$old537$,'')))<>length($old537$  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.round_id=p_round_id AND l.call_identity IS NOT NULL
    AND l.entry_kind IN('reservation','dispatch') AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
      WHERE later.call_identity=l.call_identity AND later.entry_id>l.entry_id))$old537$) THEN
  RAISE EXCEPTION 'recovery round call head source differs' USING ERRCODE='55000';
 END IF;
 EXECUTE replace(d,$old537$  OR EXISTS(SELECT 1 FROM public.lab_arena_ledger l WHERE l.round_id=p_round_id AND l.call_identity IS NOT NULL
    AND l.entry_kind IN('reservation','dispatch') AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
      WHERE later.call_identity=l.call_identity AND later.entry_id>l.entry_id))$old537$,$new537$  OR EXISTS (
    -- Reuse533's round-local latest heads before the global CID anti-probe.
    WITH round_call_rows AS MATERIALIZED (
      SELECT local.call_identity,local.entry_id,local.entry_kind
      FROM public.lab_arena_ledger AS local
      WHERE local.round_id=p_round_id AND local.call_identity IS NOT NULL
    )
    SELECT 1 FROM (
      SELECT DISTINCT ON (local.call_identity)
        local.call_identity,local.entry_id,local.entry_kind
      FROM round_call_rows AS local
      ORDER BY local.call_identity,local.entry_id DESC
    ) AS opened
    WHERE opened.entry_kind IN ('reservation','dispatch')
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger AS later
        WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))$new537$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'b66646eeb66663214623d5771feb60a03f6ce9821541341d2a68bb2423c02b84' THEN
  RAISE EXCEPTION 'recovery round call head readback differs' USING ERRCODE='55000';
 END IF;
END;
$recovery_call_heads537$;
COMMIT;
