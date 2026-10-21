-- Prune closed round-local call heads before the unchanged global head guard.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $round_call_heads$
DECLARE d TEXT; metadata JSONB; target OID;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
    WHERE i.indexrelid=to_regclass('public.lab_arena_ledger_round_call_head_idx')
      AND i.indrelid='public.lab_arena_ledger'::regclass AND i.indisvalid AND i.indisready AND i.indislive
      AND pg_get_userbyid(c.relowner)='lab_arena_owner'
      AND pg_get_indexdef(i.indexrelid)='CREATE INDEX lab_arena_ledger_round_call_head_idx ON public.lab_arena_ledger USING btree (round_id) WHERE (call_identity IS NOT NULL)') THEN
    RAISE EXCEPTION 'round call head index differs' USING ERRCODE='55000';
  END IF;
  target:='public.lab_arena_cancel_round(text,text)'::regprocedure;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')<>'c984c3ef7cb6f8ab8970b152e44336bc7eb3135a9e31ebda8f00c331c7aa1700' THEN
    IF encode(extensions.digest(d,'sha256'),'hex')<>'befdc48d4bf66284c0bdb45861caad294ad736060290eaa06c1b115191092f29'
      OR (length(d)-length(replace(d,$old$    OR EXISTS (SELECT 1 FROM public.lab_arena_ledger AS opened
      WHERE opened.round_id=p_round_id
        AND opened.call_identity IS NOT NULL
        AND opened.entry_kind IN ('reservation','dispatch')
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger AS later
          WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))$old$,'')))<>length($old$    OR EXISTS (SELECT 1 FROM public.lab_arena_ledger AS opened
      WHERE opened.round_id=p_round_id
        AND opened.call_identity IS NOT NULL
        AND opened.entry_kind IN ('reservation','dispatch')
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger AS later
          WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))$old$) THEN
      RAISE EXCEPTION 'round call head cancel source differs' USING ERRCODE='55000';
    END IF;
    EXECUTE replace(d,$old$    OR EXISTS (SELECT 1 FROM public.lab_arena_ledger AS opened
      WHERE opened.round_id=p_round_id
        AND opened.call_identity IS NOT NULL
        AND opened.entry_kind IN ('reservation','dispatch')
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger AS later
          WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))$old$,$new$    OR EXISTS (
      -- Materialize only this round's narrow rows before ordering call heads.
      -- Otherwise an ordered global CID scan can read every other round.
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
          WHERE later.call_identity=opened.call_identity AND later.entry_id>opened.entry_id))$new$);
  END IF;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'c984c3ef7cb6f8ab8970b152e44336bc7eb3135a9e31ebda8f00c331c7aa1700' THEN
    RAISE EXCEPTION 'round call head cancel readback differs' USING ERRCODE='55000';
  END IF;
END;
$round_call_heads$;
COMMIT;
