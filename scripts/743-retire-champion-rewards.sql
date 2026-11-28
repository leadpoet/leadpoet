-- A signed epoch can switch to fixed-quote weights before any job is payable.
-- The operator activates this prospectively by inserting the signed predecessor.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path = pg_catalog;

CREATE TEMP TABLE retirement_schema_acl ON COMMIT DROP AS
 SELECT nspacl, has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_champion_retirement (
 singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
 first_epoch bigint NOT NULL CHECK (first_epoch > 0),
 previous_state_hash text NOT NULL CHECK (previous_state_hash ~ '^sha256:[0-9a-f]{64}$')
);
ALTER TABLE public.collection_champion_retirement OWNER TO lab_arena_owner;
ALTER TABLE public.collection_champion_retirement ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.collection_champion_retirement FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE FUNCTION public.collection_champion_retirement_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE prior public.lab_arena_accepted_weight_states;
BEGIN
 IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'champion retirement is immutable'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
 SELECT * INTO prior FROM public.lab_arena_accepted_weight_states
 WHERE network='finney' AND netuid=71 ORDER BY epoch DESC LIMIT 1;
 IF NOT FOUND OR NEW.first_epoch<>prior.epoch+1 OR NEW.previous_state_hash<>prior.state_hash THEN
  RAISE EXCEPTION 'champion retirement must follow the latest signed epoch'; END IF;
 IF EXISTS (SELECT FROM public.collection_reward_cutover) THEN
  RAISE EXCEPTION 'champion emissions already replaced by paid collection'; END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER immutable_champion_retirement BEFORE INSERT OR UPDATE OR DELETE ON public.collection_champion_retirement
 FOR EACH ROW EXECUTE FUNCTION public.collection_champion_retirement_guard_v1();
CREATE TRIGGER no_truncate_champion_retirement BEFORE TRUNCATE ON public.collection_champion_retirement
 FOR EACH STATEMENT EXECUTE FUNCTION public.collection_champion_retirement_guard_v1();
ALTER FUNCTION public.collection_champion_retirement_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_champion_retirement_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Existing gateways use this boundary for champion admission and weight routing.
-- Funded jobs and prices continue to use collection_reward_schedule_v1().
CREATE OR REPLACE FUNCTION public.collection_reward_start_v1() RETURNS jsonb
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT jsonb_build_object('first_epoch', least(
   (SELECT first_epoch FROM public.collection_champion_retirement),
   (SELECT first_epoch FROM public.collection_reward_cutover)))
$$;

CREATE OR REPLACE FUNCTION public.collection_reward_state_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE first_weight bigint; first_paid bigint; expected jsonb; basis jsonb := NEW.state_doc->'reward_basis';
BEGIN
 IF NEW.network<>'finney' OR NEW.netuid<>71 THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 first_weight := (public.collection_reward_start_v1()->>'first_epoch')::bigint;
 IF first_weight IS NULL OR NEW.epoch<first_weight THEN
  IF basis->>'schema_version'='leadpoet.collection.fixed_run_quotes.v1' THEN
   RAISE EXCEPTION 'premature collection payout'; END IF;
  RETURN NEW;
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_epoch:'||NEW.epoch::text,0));
 expected := public.collection_reward_quotes_v1(NEW.epoch);
 IF (basis-'valuation') IS DISTINCT FROM expected OR NOT (basis ? 'valuation') THEN
  RAISE EXCEPTION 'collection epoch quote mismatch'; END IF;
 SELECT c.first_epoch INTO first_paid FROM public.collection_reward_cutover c;
 IF first_paid IS NULL OR NEW.epoch<first_paid THEN
  IF expected->>'job_count'<>'0' OR basis->'valuation' IS DISTINCT FROM 'null'::jsonb
     OR NEW.state_doc->>'burn_hotkey' IS DISTINCT FROM (
        SELECT w.state_doc->>'burn_hotkey' FROM public.collection_champion_retirement r
        JOIN public.lab_arena_accepted_weight_states w ON w.network='finney' AND w.netuid=71
        AND w.epoch=r.first_epoch-1 AND w.state_hash=r.previous_state_hash) THEN
   RAISE EXCEPTION 'retired epoch requires the original burn hotkey and zero payable work'; END IF;
 END IF;
 RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION public.collection_legacy_round_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF NEW.configuration_doc->>'mode' IS DISTINCT FROM 'live' OR
    NEW.configuration_doc->>'network_name' IS DISTINCT FROM 'finney' OR
    NEW.configuration_doc->>'netuid' IS DISTINCT FROM '71' THEN RETURN NEW; END IF;
 IF TG_OP='UPDATE' AND (NEW.promotion_doc,NEW.reward_activated_at)
     IS NOT DISTINCT FROM (OLD.promotion_doc,OLD.reward_activated_at) THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 IF public.collection_reward_start_v1()->>'first_epoch' IS NOT NULL THEN
  RAISE EXCEPTION 'retired champion competition'; END IF;
 RETURN NEW;
END $$;

-- Fence an old gateway or an upload whose request started before retirement.
CREATE FUNCTION public.collection_retired_submission_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF TG_OP='UPDATE' AND (OLD.status IS DISTINCT FROM 'uploading' OR NEW.status IS DISTINCT FROM 'accepted') THEN
  RETURN NEW;
 END IF;
 IF EXISTS (SELECT FROM public.lab_arena_rounds WHERE round_id=NEW.round_id AND
     configuration_doc->>'mode'='live' AND configuration_doc->>'network_name'='finney'
     AND configuration_doc->>'netuid'='71') THEN
  PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
  IF public.collection_reward_start_v1()->>'first_epoch' IS NOT NULL THEN
   RAISE EXCEPTION 'retired champion submissions'; END IF;
 END IF;
 RETURN NEW;
END $$;
ALTER FUNCTION public.collection_retired_submission_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.collection_retired_submission_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE TRIGGER collection_retire_submissions BEFORE INSERT OR UPDATE OF status ON public.lab_arena_submissions
 FOR EACH ROW EXECUTE FUNCTION public.collection_retired_submission_guard_v1();

-- A later paid launch still requires its normal signed anchor, pilot evidence,
-- and first accepted live job. Retired competition records can stay historical.
CREATE OR REPLACE FUNCTION public.collection_reward_cutover_guard_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE last_signed bigint;
BEGIN
 IF TG_OP <> 'INSERT' THEN RAISE EXCEPTION 'collection cutover is immutable'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_reward_cutover',0));
 SELECT max(epoch) INTO last_signed FROM public.lab_arena_accepted_weight_states
 WHERE network=NEW.network AND netuid=NEW.netuid;
 IF last_signed IS NULL OR NEW.first_epoch<>last_signed+1 THEN
  RAISE EXCEPTION 'collection cutover must immediately follow the latest signed epoch'; END IF;
 IF EXISTS (SELECT FROM public.collection_champion_retirement) THEN RETURN NEW; END IF;
 IF EXISTS(SELECT FROM public.lab_arena_rounds r WHERE r.configuration_doc->>'network_name'='finney'
           AND r.configuration_doc->>'netuid'='71' AND r.configuration_doc->>'mode'='live'
           AND ((r.rewards_enabled AND r.status NOT IN ('published','cancelled')) OR
           (r.promotion_doc IS NOT NULL AND r.baseline_promoted_at IS NULL)
           OR r.effective_reward_epoch >= NEW.first_epoch)) THEN
  RAISE EXCEPTION 'legacy competition or prepared promotion must finish before cutover'; END IF;
 IF (SELECT r.reward_activated_at IS NULL FROM public.lab_arena_rounds r
     WHERE r.configuration_doc->>'network_name'='finney' AND r.configuration_doc->>'netuid'='71'
       AND r.configuration_doc->>'mode'='live' AND r.rewards_enabled AND r.status='published'
     ORDER BY r.evaluation_date DESC NULLS LAST,r.published_at DESC LIMIT 1) THEN
  RAISE EXCEPTION 'latest legacy award must finish before cutover'; END IF;
 RETURN NEW;
END $$;

DO $$ BEGIN
 IF NOT (SELECT had_create FROM pg_temp.retirement_schema_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
    (SELECT nspacl FROM pg_temp.retirement_schema_acl) THEN RAISE EXCEPTION 'unexpected schema privileges'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
