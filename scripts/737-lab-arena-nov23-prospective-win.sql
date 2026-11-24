-- An owner-authorized replay may earn future epochs after a later no-win round.
-- Keep its published successor, activated bases and accepted states immutable.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
SELECT pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
CREATE TEMP TABLE arena737_schema ON COMMIT DROP AS
SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $installed$
DECLARE f pg_proc;
BEGIN
 SELECT * INTO f FROM pg_proc WHERE oid=to_regprocedure('public.lab_arena_prospective_win_737(text)');
 IF FOUND AND (encode(extensions.digest(f.prosrc,'sha256'),'hex')<>'3889f4f657bef687cce84f0c004ece6446071629515d8a1beda44ee192f9d73d'
   OR pg_get_userbyid(f.proowner)<>'lab_arena_owner' OR NOT f.prosecdef OR f.provolatile<>'s'
   OR f.proacl::text IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
   OR f.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[]) THEN
   RAISE EXCEPTION '737 installed authority differs';
 END IF;
END
$installed$;

CREATE OR REPLACE FUNCTION public.lab_arena_prospective_win_737(p_round_id text)
RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $guard$
 SELECT EXISTS (
   SELECT 1 FROM public.lab_arena_scoring_supersessions h
   JOIN public.lab_arena_rounds r ON r.round_id=h.round_id
   JOIN public.lab_arena_rounds n ON n.round_id=h.recovery_doc->>'later_round_id'
   WHERE p_round_id='arena-2026-11-23' AND h.round_id=p_round_id
     AND h.request_id='nov23-restore-unclaimed-736-v1'
     AND h.actor_ref='authorized-nov23-dispatch-recovery'
     AND h.recovery_doc->>'operation'='round_operational_update_v1'
     AND h.recovery_doc->>'reason'='nov23_dispatch_after_predecessor'
     AND h.old_round_doc->>'cancel_reason'='scoring_incomplete'
     AND h.old_round_doc->>'status'='cancelled' AND h.new_round_doc->>'status'='committed'
     AND r.evaluation_date='2026-11-23' AND r.status<>'cancelled'
     AND r.configuration_doc->>'mode'='live' AND r.rewards_enabled
     AND r.arena_network_name='finney' AND r.arena_netuid=71
     AND n.round_id='arena-2026-11-24' AND n.evaluation_date='2026-11-24'
     AND n.status='published' AND n.king_outcome='no_king'
     AND n.publication_doc#>>'{king_decision,outcome}'='no_king'
     AND n.reward_activated_at IS NOT NULL AND n.signing_key_doc IS NOT NULL
     AND n.reward_basis_hash='sha256:94f99614eaed7e18b7e8114dc9e69b1a5c03fa2dddbe436d390216b94939abe7'
     AND n.reward_basis_hash=h.recovery_doc->>'later_reward_basis_hash'
     AND n.reward_basis_hash=n.reward_basis_doc->>'reward_basis_hash'
     AND (r.reward_activated_at IS NOT NULL OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_rounds future
       WHERE future.arena_network_name=r.arena_network_name AND future.arena_netuid=r.arena_netuid
       AND future.configuration_doc->>'mode'='live' AND future.rewards_enabled
       AND coalesce(future.evaluation_date, to_char(
           (future.configuration_doc#>>'{schedule,submission_cutoff}')::timestamptz AT TIME ZONE 'UTC',
           'YYYY-MM-DD'))>n.evaluation_date
       AND (future.champion_funding_frozen OR future.status NOT IN ('open','cancelled'))
     ))
 )
$guard$;
ALTER FUNCTION public.lab_arena_prospective_win_737(text) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prospective_win_737(text) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_prospective_win_737(text) TO lab_arena_service;
DO $acl$
BEGIN
 IF NOT (SELECT had_create FROM arena737_schema) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM arena737_schema) THEN
   RAISE EXCEPTION '737 schema ACL differs'; END IF;
END
$acl$;
CREATE TEMP TABLE arena737_patch(sig text, old_hash text,new_hash text,old_part text,new_part text,idx integer) ON COMMIT DROP;
INSERT INTO arena737_patch VALUES
('public.lab_arena_prepare_promotion(text,jsonb)','5919a7bf00b2e7943ed18f4e83e1a30abb3a2cec0da592c1e1b8909da9a76c96','1feb27740a4186b9f218c8cbfeec00716657fb41596a0c85e6a93f4999b044e2',$old$IF v_newer_round_id IS NOT NULL THEN$old$,$new$IF v_newer_round_id IS NOT NULL AND NOT (
    v_newer_round_id='arena-2026-11-24'
    AND v_round.publication_doc#>>'{king_decision,outcome}'='crowned'
    AND public.lab_arena_prospective_win_737(p_round_id)
  ) THEN$new$,0),
('public.lab_arena_activate_reward(text,jsonb,jsonb)','c1e3c1aa7d848c06bf9f49fcef03d99e642af8dd6ec1530447387fb1bbc398c1','bb0562520af8293f238a1dcc6e21037427e83f7a7fac4566ba2ae1d480c3295e',$old$IF v_newer_round_id IS NOT NULL THEN$old$,$new$IF v_newer_round_id IS NOT NULL AND NOT (
    v_newer_round_id='arena-2026-11-24'
    AND v_round.publication_doc#>>'{king_decision,outcome}'='crowned'
    AND public.lab_arena_prospective_win_737(p_round_id)
  ) THEN$new$,0),
('public.lab_arena_activate_reward(text,jsonb,jsonb)','c1e3c1aa7d848c06bf9f49fcef03d99e642af8dd6ec1530447387fb1bbc398c1','bb0562520af8293f238a1dcc6e21037427e83f7a7fac4566ba2ae1d480c3295e',$old$  IF v_maximum IS NOT NULL AND v_effective <= v_maximum THEN$old$,$new$  -- A replay can only introduce a future signed state.
  IF p_round_id='arena-2026-11-23' AND public.lab_arena_prospective_win_737(p_round_id)
    AND EXISTS (SELECT 1 FROM public.lab_arena_accepted_weight_states s
      WHERE s.network=v_round.arena_network_name AND s.netuid=v_round.arena_netuid AND s.epoch>=v_effective)
  THEN RAISE EXCEPTION '737 reward epoch already has a signed state'; END IF;
  IF v_maximum IS NOT NULL AND v_effective <= v_maximum THEN$new$,1),
('public.lab_arena_reward_slot_snapshot(text,jsonb)','6f9ea6982166fd7818d3bd249ff46777d8d9e2229a9d8ed85f5977eff0510765','c04e701cc5d6a079f1d22c19e38a3cf1e2e83150a6c875a3bf06463d0652da9f',$old$AND prior.evaluation_date <= v_round.evaluation_date$old$,$new$AND (prior.evaluation_date <= v_round.evaluation_date
        OR (prior.round_id='arena-2026-11-24' AND public.lab_arena_prospective_win_737(p_round_id)))$new$,0),
('public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)','4db417c9709302922407a7a0690093bfb23bf467c61f72b441b9fa7a558e5e58','4ca96a42e4cb11726ed19673a2c96786c58bdc4cc7cf565a06b1225aed102c87',$old$  IF evaluation_day IS NULL THEN RETURN FALSE; END IF;$old$,$new$  IF evaluation_day IS NULL THEN RETURN FALSE; END IF;
  IF evaluation_day>'2026-11-24' AND p_round.arena_network_name='finney' AND p_round.arena_netuid=71
    AND p_round.rewards_enabled AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds r WHERE r.round_id='arena-2026-11-23'
        AND r.reward_activated_at IS NULL
        AND (r.status NOT IN ('published','cancelled') OR r.king_outcome='crowned')
        AND public.lab_arena_prospective_win_737(r.round_id)
    ) THEN RETURN FALSE; END IF;$new$,0);
DO $patch$
DECLARE target record; fragment record; source text; meta jsonb; id oid;
BEGIN
 FOR target IN SELECT DISTINCT sig,old_hash,new_hash FROM arena737_patch LOOP
   id:=target.sig::regprocedure;
   SELECT pg_get_functiondef(id),to_jsonb(p)-'prosrc' INTO source,meta FROM pg_proc p WHERE p.oid=id;
   IF (SELECT encode(extensions.digest(prosrc,'sha256'),'hex') FROM pg_proc WHERE oid=id)=target.new_hash THEN CONTINUE; END IF;
   IF (SELECT encode(extensions.digest(prosrc,'sha256'),'hex') FROM pg_proc WHERE oid=id)<>target.old_hash
      OR pg_get_userbyid((meta->>'proowner')::oid)<>'lab_arena_owner' OR NOT (meta->>'prosecdef')::bool THEN RAISE EXCEPTION '737 function preimage differs: %',target.sig; END IF;
   FOR fragment IN SELECT * FROM arena737_patch WHERE sig=target.sig ORDER BY idx LOOP
     source:=replace(source,fragment.old_part,fragment.new_part);
   END LOOP;
   EXECUTE source;
   IF NOT EXISTS (SELECT FROM pg_proc p WHERE p.oid=id AND to_jsonb(p)-'prosrc'=meta
      AND encode(extensions.digest(prosrc,'sha256'),'hex')=target.new_hash) THEN RAISE EXCEPTION '737 function readback differs: %',target.sig; END IF;
 END LOOP;
END
$patch$;
NOTIFY pgrst,'reload schema';
COMMIT;
