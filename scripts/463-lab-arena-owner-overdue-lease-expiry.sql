-- Let the authenticated restart owner use ordinary expiry for overdue,
-- captured execute and score work. Preserve dispatch liability and immutable evidence.
-- No lease extension, result recovery, guard edit, or early open-call expiry.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $guarded_overdue_lease463$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_expire_leases(text)';
  v_preimage CONSTANT TEXT := '6fb3d102cf53f32ec2cdc653d3a43af398897019a16db83885136defa96b17bb';
  v_postimage CONSTANT TEXT := '11a47cbe7ba950bdc284342a2ede73cd08aba7b0f55e452131d3bf87c9a81c15';
  v_definition TEXT;
  v_identity JSONB;
  v_old CONSTANT TEXT := $old$    -- Preserve404: a whole-round call cannot naturally expire in-flight work.
    IF EXISTS (SELECT 1 FROM public.lab_arena_runs candidate
      WHERE candidate.round_id=p_round_id AND candidate.status='leased'
        AND candidate.lease_expires_at<=clock_timestamp()
        AND (EXISTS (SELECT 1 FROM (
          SELECT DISTINCT ON (cost.call_identity) cost.entry_kind
          FROM public.lab_arena_ledger cost WHERE cost.run_id=candidate.run_id
            AND cost.call_identity IS NOT NULL
          ORDER BY cost.call_identity,cost.entry_id DESC
        ) heads WHERE heads.entry_kind IN ('reservation','dispatch'))
        OR EXISTS (SELECT 1 FROM public.lab_arena_ledger cost
          WHERE cost.run_id=candidate.run_id AND cost.call_identity IS NULL))) THEN
      RETURN jsonb_build_object('status','ok','expired',0,'retried',0);
    END IF;
$old$;
  v_new CONSTANT TEXT := $new$    -- lab_arena_guard_owned_overdue_expiry_v1: the exact owner,
    -- capture, stage and result fences above retain the whole-round write set.
    -- Only already overdue work may close open calls through the
    -- unchanged ordinary expiry tail. Dispatch remains full uncertainty;
    -- unidentified or unbound accounting remains held.
    IF EXISTS (SELECT 1 FROM public.lab_arena_runs candidate
      WHERE candidate.round_id=p_round_id AND candidate.status='leased'
        AND (EXISTS (SELECT 1 FROM (
          SELECT DISTINCT ON (cost.call_identity) cost.*
          FROM public.lab_arena_ledger cost WHERE cost.run_id=candidate.run_id
            AND cost.call_identity IS NOT NULL
          ORDER BY cost.call_identity,cost.entry_id DESC
        ) heads WHERE heads.entry_kind IN ('reservation','dispatch')
          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger reservation
            WHERE reservation.call_identity=heads.call_identity
              AND reservation.entry_kind='reservation'
              AND reservation.run_id=candidate.run_id
              AND reservation.round_id=candidate.round_id
              AND reservation.submission_id=candidate.submission_id
              AND reservation.miner_hotkey=candidate.miner_hotkey
              AND reservation.stage=candidate.stage
              AND reservation.provider=heads.provider
              AND reservation.operation_id=heads.operation_id
              AND reservation.funding_source=heads.funding_source
              AND reservation.amount_microusd=heads.amount_microusd
              AND heads.round_id=candidate.round_id
              AND heads.submission_id=candidate.submission_id
              AND heads.miner_hotkey=candidate.miner_hotkey
              AND heads.stage=candidate.stage))
        OR EXISTS (SELECT 1 FROM public.lab_arena_ledger cost
          WHERE cost.run_id=candidate.run_id AND (cost.call_identity IS NULL
            OR cost.call_identity !~ '^sha256:[0-9a-f]{64}$')))) THEN
      RETURN jsonb_build_object('status','ok','expired',0,'retried',0);
    END IF;
$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::text,
    p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_signature::regprocedure;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'Arena overdue expiry security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (length(v_definition)-length(replace(v_definition,v_old,'')))/length(v_old)<>1 THEN
    RAISE EXCEPTION 'Arena overdue expiry preimage differs' USING ERRCODE='55000';
  END IF;
  v_definition:=replace(v_definition,v_old,v_new);
  IF encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena overdue expiry postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF encode(extensions.digest(pg_get_functiondef(v_signature::regprocedure),'sha256'),'hex')<>v_postimage
    OR (SELECT jsonb_build_array(owner.rolname,p.proacl::text,p.prosecdef,p.provolatile,p.proconfig)
      FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
      WHERE p.oid=v_signature::regprocedure) IS DISTINCT FROM v_identity THEN
    RAISE EXCEPTION 'Arena overdue expiry readback differs' USING ERRCODE='55000';
  END IF;
END;
$guarded_overdue_lease463$;
NOTIFY pgrst,'reload schema';
COMMIT;
