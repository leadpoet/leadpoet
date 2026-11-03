-- Freeze verified champion provider fallback into new scoring claims.
-- Preserve old score payer provenance, accepted work, execution restart rules,
-- credential evidence, budgets, billing identities, and the half reward factor.
-- Deploy the matching broker first; it also supports the previous SQL schema.
-- After this migration, retain that broker when rolling back other changes:
-- older brokers cannot resolve host-funded score compatibility routes safely.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $champion_score_funding576_0$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_provider_funding(text,text)';
  preimage CONSTANT TEXT := 'b83c135347d6ea80fb3e21ebbc5bbf992afc87a5e04ed6404a4831707ef3d03d';
  postimage CONSTANT TEXT := '32fada1e9d9b9f812a7d8a0ff8e0d013b84d8bbaf481eaf055fc84d8e94fb51a';
  d TEXT; metadata JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=signature::regprocedure
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='s'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN
    RAISE EXCEPTION 'champion score funding preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old0$-- Execution keeps its immutable per-run fallback snapshot. Scoring always
    -- remains on the promoted miner account and cannot inherit host fallback.$old0$,''))) <> length($old0$-- Execution keeps its immutable per-run fallback snapshot. Scoring always
    -- remains on the promoted miner account and cannot inherit host fallback.$old0$) THEN
    RAISE EXCEPTION 'champion score funding anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old0$-- Execution keeps its immutable per-run fallback snapshot. Scoring always
    -- remains on the promoted miner account and cannot inherit host fallback.$old0$,$new0$-- Both kinds keep an immutable payer. Older score runs without a
    -- snapshot retain their original promoted-miner payer.$new0$);
  IF (length(d)-length(replace(d,$old1$WHEN v_champion AND v_run.kind = 'score' THEN 'miner_key'$old1$,''))) <> length($old1$WHEN v_champion AND v_run.kind = 'score' THEN 'miner_key'$old1$) THEN
    RAISE EXCEPTION 'champion score funding anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old1$WHEN v_champion AND v_run.kind = 'score' THEN 'miner_key'$old1$,$new1$WHEN v_champion AND v_run.kind = 'score' THEN
      CASE WHEN v_run.champion_funding_sources IS NULL THEN 'miner_key'
        ELSE v_run.champion_funding_sources ->> p_provider END$new1$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'champion score funding postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=signature::regprocedure)
      IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'champion score funding readback differs' USING ERRCODE='55000';
  END IF;
END;
$champion_score_funding576_0$;

DO $champion_score_funding576_1$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  preimage CONSTANT TEXT := 'a01a0cd11bba2abb77f74d04ee780cab32a04ff2c08eabbb448b90ae280e92f7';
  postimage CONSTANT TEXT := '11f8669d16966fbe41e5c2530c3ed273fea2abc1aacfc4264e2854f017f5b6f3';
  d TEXT; metadata JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=signature::regprocedure
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='v'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN
    RAISE EXCEPTION 'champion score funding preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old0$WHEN v_run.kind = 'execute' AND v_submission.is_king$old0$,''))) <> length($old0$WHEN v_run.kind = 'execute' AND v_submission.is_king$old0$) THEN
    RAISE EXCEPTION 'champion score funding anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old0$WHEN v_run.kind = 'execute' AND v_submission.is_king$old0$,$new0$WHEN v_run.kind IN ('execute','score') AND v_submission.is_king$new0$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'champion score funding postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=signature::regprocedure)
      IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'champion score funding readback differs' USING ERRCODE='55000';
  END IF;
END;
$champion_score_funding576_1$;

DO $champion_score_funding576_2$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)';
  preimage CONSTANT TEXT := 'b800eaf326ae18e3090fbeb35447c7f516db5104782750bdb9d64b7af1803104';
  postimage CONSTANT TEXT := '3ebdca78d95dfb198a0e918dd1bbe65b9356fb8afbc9aaa42814a3e43ee85999';
  d TEXT; metadata JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=signature::regprocedure
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='v'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')=postimage THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>preimage THEN
    RAISE EXCEPTION 'champion score funding preimage differs' USING ERRCODE='55000';
  END IF;
  IF (length(d)-length(replace(d,$old0$IF v_run.champion_restart_required OR
       (v_run.champion_funding_sources ->> p_provider = 'miner_key'
        AND p_provider = ANY(v_round.champion_fallback_providers)) THEN$old0$,''))) <> length($old0$IF v_run.champion_restart_required OR
       (v_run.champion_funding_sources ->> p_provider = 'miner_key'
        AND p_provider = ANY(v_round.champion_fallback_providers)) THEN$old0$) THEN
    RAISE EXCEPTION 'champion score funding anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old0$IF v_run.champion_restart_required OR
       (v_run.champion_funding_sources ->> p_provider = 'miner_key'
        AND p_provider = ANY(v_round.champion_fallback_providers)) THEN$old0$,$new0$IF v_run.kind = 'execute' AND (v_run.champion_restart_required OR
       (v_run.champion_funding_sources ->> p_provider = 'miner_key'
        AND p_provider = ANY(v_round.champion_fallback_providers))) THEN$new0$);
  IF (length(d)-length(replace(d,$old1$IF v_run.champion_funding_sources IS NOT NULL THEN$old1$,''))) <> length($old1$IF v_run.champion_funding_sources IS NOT NULL THEN$old1$) THEN
    RAISE EXCEPTION 'champion score funding anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$old1$IF v_run.champion_funding_sources IS NOT NULL THEN$old1$,$new1$IF v_run.kind = 'execute' AND v_run.champion_funding_sources IS NOT NULL THEN$new1$);
  IF encode(extensions.digest(d,'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'champion score funding postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE d;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=signature::regprocedure)
      IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')<>postimage THEN
    RAISE EXCEPTION 'champion score funding readback differs' USING ERRCODE='55000';
  END IF;
END;
$champion_score_funding576_2$;
COMMIT;
