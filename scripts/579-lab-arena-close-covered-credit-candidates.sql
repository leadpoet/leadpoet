-- Retain the existing recovery guard, but eliminate already-covered score
-- attempts before its full credit proof. This avoids repeated obsolete proof
-- scans while the claim-control advisory lock is held. The global accepted
-- anti-join, proof, deadline, active lease, baseline, floor and lock rules stay
-- unchanged. No data, index, funding, timeout or publication rule changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $close_covered_credit579$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_close_scoring(text,smallint)';
  definition TEXT; updated TEXT; security_before JSONB; security_after JSONB;
BEGIN
  IF encode(extensions.digest(pg_get_functiondef(to_regprocedure(
      'public.lab_arena_miner_credit_failure_v1(text)')),'sha256'),'hex')
      IS DISTINCT FROM '23fbb6e5d40450916b75110d16d9b3c9c369264014dea51e17db7f3f978786c6' THEN
    RAISE EXCEPTION 'Close covered credit helper preimage differs';
  END IF;
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(o.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,security_before
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF definition IS NULL OR security_before IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public'],'plpgsql') THEN
    RAISE EXCEPTION 'Close covered credit security preimage differs';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='10093875bac014642baa5056bc1f578611c46a327b9b7e870a9a2ca454c5d636' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM
      'a07b809f89df827f3e0034e71f01c587b49bf1718715c7b60320e1521f322a6c' THEN
    RAISE EXCEPTION 'Close covered credit exact preimage differs';
  END IF;
  updated:=replace(definition,$old_credit579$      AND EXISTS (SELECT 1 FROM public.lab_arena_runs credit WHERE credit.round_id=p_round_id
        AND credit.stage=p_stage AND credit.kind='score'
        AND public.lab_arena_miner_credit_failure_v1(credit.run_id)
        AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs accepted
          WHERE accepted.scored_run_id=credit.scored_run_id AND accepted.kind='score'
            AND accepted.status='accepted'))$old_credit579$,
    $new_credit579$      AND EXISTS (
        WITH uncovered_credit AS MATERIALIZED (
          SELECT credit.run_id FROM public.lab_arena_runs credit WHERE credit.round_id=p_round_id
            AND credit.stage=p_stage AND credit.kind='score'
            AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs accepted
              WHERE accepted.scored_run_id=credit.scored_run_id AND accepted.kind='score'
                AND accepted.status='accepted')
        )
        SELECT 1 FROM uncovered_credit credit
        WHERE public.lab_arena_miner_credit_failure_v1(credit.run_id))$new_credit579$);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM
      '10093875bac014642baa5056bc1f578611c46a327b9b7e870a9a2ca454c5d636' THEN
    RAISE EXCEPTION 'Close covered credit target differs';
  END IF;
  EXECUTE updated;
  SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO security_after
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF encode(extensions.digest(pg_get_functiondef(signature::regprocedure),'sha256'),'hex')
      IS DISTINCT FROM '10093875bac014642baa5056bc1f578611c46a327b9b7e870a9a2ca454c5d636' OR security_after IS DISTINCT FROM security_before THEN
    RAISE EXCEPTION 'Close covered credit readback differs';
  END IF;
END;
$close_covered_credit579$;
NOTIFY pgrst,'reload schema';
COMMIT;
