-- Freeze one prospective mainnet cutover. Installing the migration never activates it.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path=pg_catalog;
CREATE TEMP TABLE collection714_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Operator schedules exactly one activation via the migration authority after
-- validator roll-out and pilot pricing verification. The app can only read it.
CREATE TABLE public.collection_reward_cutover (
 network text PRIMARY KEY CHECK (network='finney'),
 netuid integer NOT NULL CHECK (netuid=71),
 first_epoch bigint NOT NULL CHECK (first_epoch>0)
);
ALTER TABLE public.collection_reward_cutover OWNER TO lab_arena_owner;
ALTER TABLE public.collection_reward_cutover ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_reward_cutover FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE INDEX collection_reward_jobs_by_epoch ON public.collection_jobs(accepted_epoch,miner_hotkey)
 WHERE mode='live' AND status='accepted';

CREATE FUNCTION public.collection_reward_start_v1() RETURNS jsonb
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('first_epoch',(SELECT first_epoch FROM public.collection_reward_cutover))
$$;

CREATE FUNCTION public.collection_reward_quotes_v1(p_epoch bigint) RETURNS jsonb
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH earned AS MATERIALIZED (
  SELECT job_id, miner_hotkey, price_microusd,artifact_sha256
  FROM public.collection_jobs
  WHERE network='finney' AND netuid=71 AND status='accepted' AND mode='live' AND accepted_epoch=p_epoch-1
 ), totals AS (
  SELECT miner_hotkey,sum(price_microusd) amount FROM earned GROUP BY miner_hotkey
 )
 SELECT jsonb_build_object('schema_version','leadpoet.collection.fixed_run_quotes.v1',
  'source_epoch',p_epoch-1,'job_count',count(*),
  'quotes_microusd',coalesce((SELECT jsonb_object_agg(miner_hotkey,amount) FROM totals),'{}'::jsonb),
  'job_set_sha256',encode(sha256(convert_to(coalesce(string_agg(
    concat_ws('|',job_id,miner_hotkey,price_microusd,artifact_sha256),E'\n' ORDER BY job_id),''),'UTF8')),'hex'))
 FROM earned
$$;

-- A signed weight is an instruction. Its USD value here is a target, not
-- a statement that Subtensor has already emitted that amount to the miner.
CREATE FUNCTION public.collection_reward_statement_v1(p_hotkey text) RETURNS jsonb
 LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH recent AS (
  SELECT accepted_epoch+1 epoch,sum(price_microusd) quote
  FROM public.collection_jobs WHERE mode='live' AND status='accepted' AND miner_hotkey=p_hotkey
  GROUP BY accepted_epoch ORDER BY accepted_epoch DESC LIMIT 100
 ), states AS (
  SELECT recent.*,w.state_hash,w.state_doc->'reward_basis' basis
  FROM recent LEFT JOIN public.lab_arena_accepted_weight_states w ON
    w.network='finney' AND w.netuid=71 AND w.epoch=recent.epoch
 ), amounts AS (
  SELECT *, (basis->'valuation'->>'emission_microusd')::numeric budget,
     (SELECT sum(value::numeric) FROM jsonb_each_text(basis->'quotes_microusd')) total
  FROM states
 )
 SELECT coalesce(jsonb_agg(jsonb_build_object(
  'epoch',epoch,'quote_microusd',quote,'state_hash',state_hash,
  'emission_microusd',budget,'total_quote_microusd',total,
  'target_microusd',CASE WHEN budget IS NULL THEN NULL ELSE floor(quote*budget/greatest(budget,total)) END
 ) ORDER BY epoch DESC),'[]') FROM amounts
$$;

CREATE FUNCTION public.collection_reward_cutover_guard_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE last_signed bigint;
BEGIN
 IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'collection cutover is immutable'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
 SELECT max(epoch) INTO last_signed FROM public.lab_arena_accepted_weight_states
 WHERE network=NEW.network AND netuid=NEW.netuid;
 IF last_signed IS NULL OR NEW.first_epoch <> last_signed+1 THEN
  RAISE EXCEPTION 'collection cutover must immediately follow the latest accepted legacy epoch';
 END IF;
 IF EXISTS(SELECT FROM public.lab_arena_rounds r WHERE r.configuration_doc->>'network_name'='finney'
           AND r.configuration_doc->>'netuid'='71' AND r.configuration_doc->>'mode'='live'
           AND r.rewards_enabled AND (r.status NOT IN ('published','cancelled') OR
           (r.promotion_doc IS NOT NULL AND r.baseline_promoted_at IS NULL)
           OR r.effective_reward_epoch >= NEW.first_epoch)) THEN
  RAISE EXCEPTION 'legacy competition or prepared promotion must finish before cutover';
 END IF;
 IF (SELECT r.reward_activated_at IS NULL FROM public.lab_arena_rounds r
     WHERE r.configuration_doc->>'network_name'='finney' AND r.configuration_doc->>'netuid'='71'
       AND r.configuration_doc->>'mode'='live' AND r.rewards_enabled AND r.status='published'
     ORDER BY r.evaluation_date DESC NULLS LAST,r.published_at DESC LIMIT 1) THEN
  RAISE EXCEPTION 'latest legacy award must finish before cutover';
 END IF;
 RETURN NEW;
END $$;

-- Fence legacy processes as well as the new gateway. A persisted preparation
-- stays pending until its exact Git update has completed and been recorded.
CREATE FUNCTION public.collection_legacy_round_guard_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF NEW.configuration_doc->>'mode' IS DISTINCT FROM 'live' OR
    NEW.configuration_doc->>'network_name' IS DISTINCT FROM 'finney' OR
    NEW.configuration_doc->>'netuid' IS DISTINCT FROM '71' THEN RETURN NEW; END IF;
 IF TG_OP='UPDATE' AND (NEW.promotion_doc,NEW.reward_activated_at)
     IS NOT DISTINCT FROM (OLD.promotion_doc,OLD.reward_activated_at) THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 IF EXISTS(SELECT FROM public.collection_reward_cutover) THEN
  RAISE EXCEPTION 'legacy collection cutover has retired competition issuance';
 END IF;
 RETURN NEW;
END $$;

CREATE FUNCTION public.collection_reward_job_guard_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE first_epoch bigint;
BEGIN
 IF TG_OP <> 'INSERT' AND OLD.mode='live' AND OLD.status='accepted' THEN
  RAISE EXCEPTION 'accepted collection job is immutable';
 END IF;
 IF TG_OP='DELETE' THEN RETURN OLD; END IF;
 IF NEW.mode='live' AND NEW.status IN ('running','accepted') THEN
  SELECT c.first_epoch INTO first_epoch FROM public.collection_reward_cutover c;
  IF first_epoch IS NULL THEN RAISE EXCEPTION 'collection payout cutover required'; END IF;
  IF NEW.status='accepted' THEN
   IF NEW.accepted_epoch < first_epoch-1 THEN RAISE EXCEPTION 'collection job predates cutover'; END IF;
   PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_epoch:'||(NEW.accepted_epoch+1)::text,0));
   IF EXISTS(SELECT FROM public.lab_arena_accepted_weight_states
             WHERE network='finney' AND netuid=71 AND epoch=NEW.accepted_epoch+1) THEN
    RAISE EXCEPTION 'collection job epoch closed: retry with current finalized epoch';
   END IF;
  END IF;
 END IF;
 RETURN NEW;
END $$;

CREATE FUNCTION public.collection_reward_state_guard_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE first_epoch bigint; expected jsonb; basis jsonb := NEW.state_doc->'reward_basis';
BEGIN
 IF NEW.network<>'finney' OR NEW.netuid<>71 THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 SELECT c.first_epoch INTO first_epoch FROM public.collection_reward_cutover c;
 IF first_epoch IS NULL OR NEW.epoch<first_epoch THEN
  IF basis->>'schema_version'='leadpoet.collection.fixed_run_quotes.v1' THEN
   RAISE EXCEPTION 'premature collection payout';
  END IF;
  RETURN NEW;
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_epoch:'||NEW.epoch::text,0));
 expected:=public.collection_reward_quotes_v1(NEW.epoch);
 IF (basis-'valuation') IS DISTINCT FROM expected OR NOT (basis ? 'valuation') THEN
  RAISE EXCEPTION 'collection epoch quote mismatch';
 END IF;
 RETURN NEW;
END $$;

CREATE TRIGGER collection_legacy_round_guard BEFORE INSERT OR UPDATE OF promotion_doc,reward_activated_at
 ON public.lab_arena_rounds FOR EACH ROW EXECUTE FUNCTION public.collection_legacy_round_guard_v1();
CREATE TRIGGER collection_reward_cutover_guard BEFORE INSERT OR UPDATE OR DELETE ON public.collection_reward_cutover
 FOR EACH ROW EXECUTE FUNCTION public.collection_reward_cutover_guard_v1();
CREATE TRIGGER collection_reward_job_guard BEFORE INSERT OR UPDATE OR DELETE ON public.collection_jobs
 FOR EACH ROW EXECUTE FUNCTION public.collection_reward_job_guard_v1();
CREATE TRIGGER collection_reward_state_guard BEFORE INSERT ON public.lab_arena_accepted_weight_states
 FOR EACH ROW EXECUTE FUNCTION public.collection_reward_state_guard_v1();

DO $acl$
DECLARE f regprocedure;
BEGIN
 FOREACH f IN ARRAY ARRAY['public.collection_reward_start_v1()'::regprocedure,
   'public.collection_reward_quotes_v1(bigint)'::regprocedure,
   'public.collection_reward_statement_v1(text)'::regprocedure,
   'public.collection_reward_cutover_guard_v1()'::regprocedure,
   'public.collection_reward_job_guard_v1()'::regprocedure,
   'public.collection_legacy_round_guard_v1()'::regprocedure,
   'public.collection_reward_state_guard_v1()'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
 GRANT EXECUTE ON FUNCTION public.collection_reward_start_v1(),public.collection_reward_quotes_v1(bigint),
 public.collection_reward_statement_v1(text)
 TO lab_arena_service;
 IF NOT (SELECT had_create FROM pg_temp.collection714_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.collection714_acl)
 THEN RAISE EXCEPTION 'collection schema ACL changed'; END IF;
END $acl$;
NOTIFY pgrst,'reload schema';
COMMIT;
