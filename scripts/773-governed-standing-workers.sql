-- Single allocator for all workers. The legacy entry point continues to service its
-- original modes and existing leases; fresh standing work requires a measured budget.
BEGIN;
SET LOCAL search_path=pg_catalog;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='90s';
SELECT pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
SELECT pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
CREATE TEMP TABLE governor_acl ON COMMIT DROP AS SELECT nspacl,
  has_schema_privilege('lab_arena_owner','public','CREATE') had_create FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE public.collection_worker_limit (
 singleton boolean PRIMARY KEY CHECK(singleton), lanes smallint NOT NULL CHECK(lanes IN (4,16,40,66)),
 changed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
INSERT INTO public.collection_worker_limit(singleton,lanes) VALUES(true,4);
CREATE TABLE public.collection_epoch_capacity (
 epoch bigint PRIMARY KEY CHECK(epoch>=0), boundary bigint NOT NULL CHECK(boundary>0),
 block_hash text NOT NULL CHECK(block_hash ~ '^0x[0-9a-f]{64}$'),
 target_limit_microusd bigint NOT NULL CHECK(target_limit_microusd>0),
 valuation jsonb NOT NULL, validated_at timestamptz NOT NULL
);
CREATE TABLE public.collection_governed_jobs (
 job_id uuid PRIMARY KEY REFERENCES public.collection_jobs, budget_epoch bigint NOT NULL REFERENCES public.collection_epoch_capacity,
 finished_epoch bigint CHECK(finished_epoch>=0), model text NOT NULL, route text NOT NULL, route_map_sha256 bytea NOT NULL, tier smallint NOT NULL, concurrent_claims smallint NOT NULL
);
CREATE INDEX collection_governed_epoch ON public.collection_governed_jobs(budget_epoch);
CREATE INDEX collection_governed_finished ON public.collection_governed_jobs(finished_epoch) WHERE finished_epoch IS NOT NULL;
CREATE INDEX collection_governed_peak ON public.collection_governed_jobs(tier,concurrent_claims);
-- Index unresolved original reservations independently of the immutable receipt/charge rows.
-- This avoids an anti-join over every historical provider call before every new claim.
CREATE TABLE public.collection_unsettled_standing_calls (
 job_id uuid NOT NULL, request_id text NOT NULL, PRIMARY KEY(job_id,request_id),
 FOREIGN KEY(job_id,request_id) REFERENCES public.collection_calls(job_id,request_id)
);
CREATE TABLE public.collection_route_backoff (
 model text NOT NULL, route text NOT NULL, unavailable_until timestamptz NOT NULL,
 observed_at timestamptz NOT NULL, delay_seconds integer NOT NULL, PRIMARY KEY(model,route)
);
DO $$ DECLARE t text; BEGIN FOREACH t IN ARRAY ARRAY['collection_worker_limit','collection_epoch_capacity',
 'collection_governed_jobs','collection_unsettled_standing_calls','collection_route_backoff'] LOOP
 EXECUTE format('ALTER TABLE public.%I OWNER TO lab_arena_owner',t);
 EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',t);
 EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',t);
END LOOP; END $$;

CREATE FUNCTION public.collection_index_pending_bill_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 IF TG_TABLE_NAME='collection_calls' THEN
  IF EXISTS(SELECT FROM public.collection_jobs WHERE job_id=NEW.job_id AND mode='live' AND batch_id LIKE 'pool-%') THEN
   INSERT INTO public.collection_unsettled_standing_calls VALUES(NEW.job_id,NEW.request_id);
  END IF;
 ELSE
  DELETE FROM public.collection_unsettled_standing_calls WHERE (job_id,request_id)=(NEW.job_id,NEW.request_id);
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER collection_index_reserved_bill AFTER INSERT ON public.collection_calls
 FOR EACH ROW WHEN(NEW.key_id IS NOT NULL) EXECUTE FUNCTION public.collection_index_pending_bill_v1();
CREATE TRIGGER collection_index_known_bill AFTER INSERT OR UPDATE OF cost_microusd ON public.collection_call_receipts
 FOR EACH ROW WHEN(NEW.cost_microusd IS NOT NULL) EXECUTE FUNCTION public.collection_index_pending_bill_v1();
INSERT INTO public.collection_unsettled_standing_calls
 SELECT c.job_id,c.request_id FROM public.collection_calls c JOIN public.collection_jobs j USING(job_id)
 LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
 WHERE j.batch_id LIKE 'pool-%' AND j.mode='live' AND c.key_id IS NOT NULL AND r.cost_microusd IS NULL;

CREATE FUNCTION public.collection_worker_lanes_v1() RETURNS integer
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT lanes::integer FROM public.collection_worker_limit WHERE singleton
$$;

CREATE FUNCTION public.collection_pool_pack_state_v1(p_batch text,p_slots integer) RETURNS text
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT CASE WHEN count(DISTINCT slot) FILTER(WHERE status IN ('accepted','rejected','failed'))=p_slots THEN 'complete'
   WHEN count(DISTINCT slot)=p_slots THEN 'running' ELSE 'open' END
 FROM public.collection_jobs WHERE batch_id=p_batch AND p_batch ~ '^pool-[a-f0-9]{64}$' AND mode='live'
$$;

-- Promotions require a genuinely overlapping group of settled jobs at the
-- prior tier. A lower operator limit takes effect for the next new lease.
CREATE FUNCTION public.collection_set_worker_lanes_v1(p_lanes integer) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE current_limit public.collection_worker_limit; BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 SELECT * INTO STRICT current_limit FROM public.collection_worker_limit WHERE singleton FOR UPDATE;
 IF p_lanes IS NULL OR p_lanes NOT IN(4,16,40,66) THEN RAISE EXCEPTION 'invalid worker tier'; END IF;
 IF p_lanes>current_limit.lanes THEN
  IF p_lanes<>(CASE current_limit.lanes WHEN 4 THEN 16 WHEN 16 THEN 40 ELSE 66 END)
   OR EXISTS(SELECT FROM public.collection_route_backoff WHERE unavailable_until>clock_timestamp())
   OR NOT EXISTS(
     SELECT FROM public.collection_governed_jobs anchor JOIN public.collection_jobs a USING(job_id)
      WHERE anchor.tier=current_limit.lanes AND anchor.concurrent_claims=current_limit.lanes
      AND a.created_at>=current_limit.changed_at AND a.status='accepted'
      AND (SELECT count(*) FROM public.collection_governed_jobs peer JOIN public.collection_jobs j USING(job_id)
        WHERE peer.tier=anchor.tier AND j.status='accepted' AND j.price_microusd>0
        AND j.created_at>=current_limit.changed_at AND j.created_at<=a.created_at AND j.finished_at>=a.created_at)>=current_limit.lanes
  ) THEN RAISE EXCEPTION 'prior worker tier lacks a settled overlapping cohort or has a route in backoff'; END IF;
 END IF;
 UPDATE public.collection_worker_limit SET lanes=p_lanes,changed_at=clock_timestamp() WHERE singleton AND lanes<>p_lanes;
 RETURN p_lanes;
END $$;

-- Private worker supplies the same finalized boundary and oracle inputs used for
-- reward valuation. A refresh can lower this epoch's ceiling, never raise it.
CREATE FUNCTION public.collection_epoch_capacity_v1(p_epoch bigint,p_boundary bigint,p_hash text,p_valuation jsonb)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE old public.collection_epoch_capacity; stamp timestamptz:=clock_timestamp(); limit_usd bigint; BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 IF p_epoch IS NULL OR p_epoch<0 OR p_boundary IS NULL OR p_boundary<=0 OR p_hash IS NULL OR p_hash !~ '^0x[0-9a-f]{64}$' THEN
  RAISE EXCEPTION 'invalid finalized funding epoch'; END IF;
 SELECT * INTO old FROM public.collection_epoch_capacity ORDER BY epoch DESC LIMIT 1;
 IF old.epoch>p_epoch OR (old.epoch=p_epoch AND (old.boundary,old.block_hash)<>(p_boundary,p_hash)) OR
   (old.epoch<p_epoch AND old.boundary>=p_boundary) THEN RAISE EXCEPTION 'standing epoch anchor mismatch'; END IF;
 IF EXISTS(SELECT FROM public.lab_arena_accepted_weight_states s WHERE s.netuid=71 AND s.network='finney' AND
   (s.epoch>p_epoch OR (s.epoch=p_epoch AND (s.state_doc->>'valid_from_block')::bigint IS DISTINCT FROM p_boundary)))
 THEN RAISE EXCEPTION 'standing epoch differs from published state'; END IF;
 IF p_valuation IS NULL THEN
  RETURN coalesce(old.epoch=p_epoch AND old.validated_at>stamp-interval '5 minutes',false);
 END IF;
 IF jsonb_typeof(p_valuation) IS DISTINCT FROM 'object' OR p_valuation->>'block_hash' IS DISTINCT FROM p_hash OR
   (p_valuation->>'block')::bigint IS DISTINCT FROM p_boundary OR EXISTS(
    SELECT FROM unnest(ARRAY['miner_alpha_rao','tao_rao_per_alpha','tao_price_microusd','price_updated_at','observed_at','emission_microusd']) fld
     WHERE coalesce(p_valuation->>fld,'') !~ '^[1-9][0-9]{0,18}$')
 OR (p_valuation->>'observed_at')::bigint NOT BETWEEN extract(epoch FROM stamp)::bigint-60 AND extract(epoch FROM stamp)::bigint+5
 OR (p_valuation->>'price_updated_at')::bigint NOT BETWEEN (p_valuation->>'observed_at')::bigint-900 AND (p_valuation->>'observed_at')::bigint
 OR (p_valuation->>'emission_microusd')::numeric IS DISTINCT FROM div(
     (p_valuation->>'miner_alpha_rao')::numeric*(p_valuation->>'tao_rao_per_alpha')::numeric*(p_valuation->>'tao_price_microusd')::numeric,1000000000000000000)
 THEN RAISE EXCEPTION 'invalid or stale emission valuation'; END IF;
 limit_usd:=trunc((p_valuation->>'emission_microusd')::numeric*0.90)::bigint;
 IF limit_usd<=0 THEN RETURN false; END IF;
 INSERT INTO public.collection_epoch_capacity(epoch,boundary,block_hash,target_limit_microusd,valuation,validated_at)
 VALUES(p_epoch,p_boundary,p_hash,limit_usd,p_valuation,stamp)
 ON CONFLICT(epoch) DO UPDATE SET target_limit_microusd=least(collection_epoch_capacity.target_limit_microusd,EXCLUDED.target_limit_microusd),
 valuation=CASE WHEN EXCLUDED.target_limit_microusd<collection_epoch_capacity.target_limit_microusd
                THEN EXCLUDED.valuation ELSE collection_epoch_capacity.valuation END,validated_at=EXCLUDED.validated_at;
 RETURN true;
END $$;

-- Include charge reservations of interrupted runs until their original native bill
-- settles. Charge across a boundary exactly once, including older active leases.
CREATE FUNCTION public.collection_epoch_liability_v1(p_epoch bigint) RETURNS bigint
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH relevant(id) AS (
  SELECT j.job_id FROM public.collection_jobs j WHERE j.mode='live' AND (j.accepted_epoch=p_epoch OR j.status IN ('running','ready'))
  UNION SELECT a.job_id FROM public.collection_governed_jobs a WHERE a.budget_epoch=p_epoch OR a.finished_epoch=p_epoch
  UNION SELECT job_id FROM public.collection_unsettled_standing_calls
 ), amounts AS (
  SELECT j.*,a.budget_epoch,a.finished_epoch,coalesce(b.spent,0) spent,coalesce(b.uncertain,0) uncertain
   FROM relevant x JOIN public.collection_jobs j ON j.job_id=x.id LEFT JOIN public.collection_governed_jobs a USING(job_id)
   LEFT JOIN LATERAL(SELECT sum(coalesce(r.cost_microusd,c.reserved_microusd)) spent,
    sum(c.reserved_microusd) FILTER(WHERE r.cost_microusd IS NULL) uncertain
    FROM public.collection_calls c LEFT JOIN public.collection_call_receipts r USING(job_id,request_id)
    WHERE j.status<>'accepted' AND c.job_id=x.id AND c.key_id IS NOT NULL) b ON true
 ) SELECT coalesce(sum(CASE
    WHEN accepted_epoch=p_epoch THEN price_microusd
    WHEN status IN ('running','ready') THEN greatest(price_microusd,ceil(greatest(budget_microusd,spent)*
     coalesce(reward_rate_bps,CASE WHEN batch_id LIKE 'pool-%' THEN 12500 ELSE 15000 END)::numeric/10000))
    WHEN status<>'accepted' AND (budget_epoch=p_epoch OR finished_epoch=p_epoch) THEN ceil(spent*coalesce(reward_rate_bps,12500)::numeric/10000)
    ELSE ceil(uncertain*coalesce(reward_rate_bps,12500)::numeric/10000) END),0)::bigint FROM amounts
$$;
CREATE FUNCTION public.collection_pool_fundable_v1(p_priority integer DEFAULT NULL) RETURNS bigint
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog AS $$
 SELECT coalesce(max(public.collection_pool_available_v1(k.key_id,NULL)),0)
 FROM public.collection_registrations r JOIN public.collection_keys k USING(registration_id)
 WHERE r.mode='live' AND k.provider='openrouter' AND k.miner_hotkey=r.miner_hotkey AND r.revoked_at IS NULL AND k.revoked_at IS NULL
 AND r.consent#>>'{body,task_scope}'='sales_sandbox'
 AND (p_priority IS NULL OR (r.consent#>>'{body,priority}')::integer=p_priority)
 AND NOT EXISTS(SELECT FROM public.collection_calls c JOIN public.collection_call_receipts b USING(job_id,request_id)
    WHERE k.key_id=c.key_id AND b.http_status IN(401,402) AND c.created_at>statement_timestamp()-interval '1 hour')
$$;

-- A crashed worker may have consumed the last available funding. Preserve the
-- original expiry behavior even when a cheap idle check suppresses all new probes.
CREATE FUNCTION public.collection_expire_worker_leases_v1(p_pool_only boolean,p_epoch bigint)
RETURNS void LANGUAGE sql VOLATILE SECURITY DEFINER SET search_path=pg_catalog AS $$
 WITH expired AS (
  UPDATE public.collection_jobs SET status='failed',finished_at=clock_timestamp(),quality='{}'
  WHERE status='running' AND lease_expires_at<=clock_timestamp() AND (NOT p_pool_only OR batch_id LIKE 'pool-%') RETURNING job_id
 ) UPDATE public.collection_governed_jobs a SET finished_epoch=greatest(a.budget_epoch,
    coalesce(p_epoch,(SELECT max(epoch) FROM public.collection_epoch_capacity),a.budget_epoch))
 FROM expired e WHERE a.job_id=e.job_id
$$;
CREATE FUNCTION public.collection_worker_funding_v1() RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('collection_claim:finney:71',0));
 IF NOT EXISTS(SELECT FROM public.lab_arena_restart_claim_control
    WHERE singleton AND NOT operator_paused AND guard_commitment='') THEN RETURN 0; END IF;
 PERFORM public.collection_expire_worker_leases_v1(true,NULL);
 RETURN public.collection_pool_fundable_v1();
END $$;

-- A route is the pinned provider route for one model. The original billing hold
-- is never released because of this scheduling signal.
CREATE FUNCTION public.collection_route_backoff_v1(p_model text,p_route text,p_wait integer DEFAULT 0)
RETURNS timestamptz LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE stamp timestamptz:=clock_timestamp(); delay integer; BEGIN
 IF p_model IS NULL OR p_model !~ '^[a-z0-9._-]+/[a-z0-9._-]+$' OR p_route IS NULL OR p_route !~ '^[a-z0-9._/-]{1,100}$'
 OR p_wait IS NULL OR p_wait NOT BETWEEN 0 AND 86400 THEN RAISE EXCEPTION 'invalid route backoff'; END IF;
 INSERT INTO public.collection_route_backoff AS h(model,route,unavailable_until,observed_at,delay_seconds)
 VALUES(p_model,p_route,stamp+greatest(30,p_wait)*interval '1 second',stamp,greatest(30,p_wait))
 ON CONFLICT(model,route) DO UPDATE SET
   delay_seconds=greatest(p_wait,CASE WHEN p_wait>0 AND h.unavailable_until>stamp THEN h.delay_seconds WHEN h.observed_at<stamp-interval '15 minutes' THEN 30 ELSE least(h.delay_seconds*2,900) END),
   unavailable_until=greatest(h.unavailable_until,stamp+greatest(p_wait,CASE WHEN p_wait>0 AND h.unavailable_until>stamp THEN h.delay_seconds WHEN h.observed_at<stamp-interval '15 minutes' THEN 30 ELSE least(h.delay_seconds*2,900) END)*interval '1 second'),
   observed_at=stamp
 RETURNING unavailable_until INTO stamp;
 RETURN stamp;
END $$;
CREATE FUNCTION public.collection_note_route_throttle_v1() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $$
DECLARE a public.collection_governed_jobs; BEGIN
 SELECT * INTO a FROM public.collection_governed_jobs WHERE job_id=NEW.job_id;
 IF FOUND AND EXISTS(SELECT FROM public.collection_calls c WHERE (c.job_id,c.request_id)=(NEW.job_id,NEW.request_id) AND c.provider='openrouter') THEN
  PERFORM public.collection_route_backoff_v1(a.model,a.route,0);
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER collection_route_throttle AFTER INSERT ON public.collection_call_receipts
 FOR EACH ROW WHEN(NEW.http_status=429) EXECUTE FUNCTION public.collection_note_route_throttle_v1();

-- Extend the canonical priced allocator rather than copying its highly scoped
-- registration, route permissions, pause, slot or cap rules into a second allocator.
DO $migration$
DECLARE fn regprocedure:='public.collection_priced_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text)'::regprocedure;
 body text; args text; marker text; addition text;
BEGIN
 SELECT p.prosrc,pg_get_function_arguments(p.oid) INTO STRICT body,args FROM pg_proc p
 WHERE p.oid=fn AND p.prosecdef AND p.proowner='lab_arena_owner'::regrole;
 marker:=' day_utc date; daily_limit integer;';
 addition:=' gov_limit bigint; open_lanes integer; tier_count integer; available bigint; normal_funds bigint;';
 IF cardinality(string_to_array(body,marker))<>2 THEN RAISE EXCEPTION 'unexpected 773 allocator declarations'; END IF;
 body:=replace(body,marker,marker||addition);
 marker:=$old$ UPDATE public.collection_jobs SET status='failed',finished_at=clock_timestamp(),quality='{}'
 WHERE status='running' AND lease_expires_at<=clock_timestamp();$old$;
 IF cardinality(string_to_array(body,marker))<>2 THEN RAISE EXCEPTION 'unexpected 773 lease expiry'; END IF;
 body:=replace(body,marker,' PERFORM public.collection_expire_worker_leases_v1(false,p_epoch);');
 marker:='   SELECT coalesce(max(slot)+1,successor_first) INTO n FROM public.collection_jobs WHERE batch_id=p_batch;';
 IF cardinality(string_to_array(body,marker))<>2 THEN RAISE EXCEPTION 'unexpected 773 slot allocator'; END IF;
 body:=replace(body,marker,$new$
   IF pooled THEN
     IF current_setting('transaction_isolation')<>'read committed' THEN RAISE EXCEPTION 'pool claims require read committed'; END IF;
     IF p_epoch IS NULL OR jsonb_typeof(p_routes) IS DISTINCT FROM 'object' THEN RETURN NULL; END IF;
     SELECT c.target_limit_microusd INTO gov_limit FROM public.collection_epoch_capacity c
      WHERE c.epoch=p_epoch AND c.validated_at>clock_timestamp()-interval '5 minutes'
      AND c.epoch=(SELECT max(epoch) FROM public.collection_epoch_capacity)
      AND NOT EXISTS(SELECT FROM public.lab_arena_accepted_weight_states s WHERE s.network='finney' AND s.netuid=71 AND s.epoch>p_epoch);
     IF NOT FOUND THEN RETURN NULL; END IF;
     IF EXISTS(SELECT FROM jsonb_array_elements_text(p_models) m(id) WHERE jsonb_typeof(p_routes->m.id) IS DISTINCT FROM 'string'
       OR p_routes->>m.id !~ '^[a-z0-9._/-]{1,100}$') THEN RETURN jsonb_build_object('error','invalid_frozen_routes'); END IF;
     IF EXISTS(SELECT FROM public.collection_jobs j JOIN public.collection_governed_jobs g USING(job_id)
        WHERE j.batch_id=p_batch AND g.route_map_sha256<>sha256(convert_to(p_routes::text,'UTF8'))) THEN
        RETURN jsonb_build_object('error','batch_routes_changed'); END IF;
     SELECT count(*) INTO open_lanes FROM public.collection_jobs WHERE mode='live' AND status='running' AND batch_id LIKE 'pool-%';
     SELECT public.collection_worker_lanes_v1() INTO tier_count;
     IF open_lanes>=tier_count THEN RETURN NULL; END IF;
     available:=least(public.collection_pool_fundable_v1(),
         trunc(greatest(0,gov_limit-public.collection_epoch_liability_v1(p_epoch))::numeric*10000/public.collection_pool_reward_rate_bps_v1())::bigint);
     normal_funds:=public.collection_pool_fundable_v1(1);
     SELECT candidate.slot INTO n FROM generate_series(0,jsonb_array_length(p_models)-1) AS candidate(slot)
     WHERE (p_budgets->>(p_models->>candidate.slot))::bigint<=available
      AND NOT EXISTS(SELECT FROM public.collection_jobs claimed WHERE claimed.batch_id=p_batch AND claimed.slot=candidate.slot)
      AND NOT EXISTS(SELECT FROM public.collection_route_backoff h WHERE h.model=p_models->>candidate.slot AND h.route=p_routes->>(p_models->>candidate.slot)
                     AND h.unavailable_until>clock_timestamp())
     ORDER BY ((p_budgets->>(p_models->>candidate.slot))::bigint<=normal_funds) DESC,
              (p_models->>candidate.slot=p_expected_model) DESC NULLS LAST,candidate.slot LIMIT 1;
     IF NOT FOUND THEN RETURN NULL; END IF;
   ELSE
$new$||marker||E'\n   END IF;');
 marker:=' RETURNING * INTO job;';
 IF cardinality(string_to_array(body,marker))<>2 THEN RAISE EXCEPTION 'unexpected 773 job insertion'; END IF;
 body:=replace(body,marker,marker||$new$
 IF pooled THEN INSERT INTO public.collection_governed_jobs(job_id,budget_epoch,model,route,route_map_sha256,tier,concurrent_claims)
   VALUES (job.job_id,p_epoch,generator,p_routes->>generator,sha256(convert_to(p_routes::text,'UTF8')),tier_count,open_lanes+1); END IF;$new$);
 EXECUTE format('CREATE FUNCTION public.collection_governed_claim_v1(%s,p_epoch bigint,p_routes jsonb) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS %L',args,body);
 EXECUTE format('CREATE OR REPLACE FUNCTION public.collection_priced_claim_v1(%s) RETURNS jsonb LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS %L',args,
   'SELECT public.collection_governed_claim_v1(p_job,p_token_hash,p_batch,p_mode,p_commit,p_scenarios,p_models,p_prices,p_budgets,p_terms_hash,p_expected_model,NULL,NULL)');
END $migration$;
-- Preserve all existing function ACLs, original native billing, and identical request
-- replays. A sibling already in flight must respect a route's backoff before spending
-- again; its previous calls and saved trace remain available.
DO $lifecycle$
DECLARE definition text; marker text; BEGIN
 SELECT pg_get_functiondef('public.collection_reserve_call_v1(uuid,text,text,text,text,bigint)'::regprocedure) INTO definition;
 marker:=' SELECT coalesce(sum(coalesce(r.cost_microusd,c.reserved_microusd)),0) INTO job_used';
 IF cardinality(string_to_array(definition,marker))<>2 THEN RAISE EXCEPTION 'unexpected governed reservation predecessor'; END IF;
 EXECUTE replace(definition,marker,$new$ IF p_provider='openrouter' AND EXISTS(
   SELECT FROM public.collection_governed_jobs a JOIN public.collection_route_backoff h USING(model,route)
   WHERE a.job_id=p_job AND h.unavailable_until>clock_timestamp()
 ) THEN RETURN jsonb_build_object('status','route_backoff'); END IF;
$new$||marker);
 SELECT pg_get_functiondef('public.collection_finish_v1(uuid,text,text,jsonb,text,bigint)'::regprocedure) INTO definition;
 marker:=$old$ RETURN jsonb_build_object('status',p_status,'job_id',p_job);$old$;
 IF cardinality(string_to_array(definition,marker))<>2 THEN RAISE EXCEPTION 'unexpected governed finish predecessor'; END IF;
 EXECUTE replace(definition,marker,$new$ IF j.batch_id LIKE 'pool-%' AND p_status IN ('failed','rejected') THEN
  UPDATE public.collection_governed_jobs SET finished_epoch=p_epoch WHERE job_id=p_job;
 END IF;
$new$||marker);
END $lifecycle$;
DO $$ DECLARE f regprocedure; BEGIN
 FOREACH f IN ARRAY ARRAY[
 'public.collection_governed_claim_v1(uuid,text,text,text,text,jsonb,jsonb,jsonb,jsonb,text,text,bigint,jsonb)'::regprocedure,
 'public.collection_worker_lanes_v1()'::regprocedure,
 'public.collection_worker_funding_v1()'::regprocedure,
 'public.collection_expire_worker_leases_v1(boolean,bigint)'::regprocedure,
 'public.collection_pool_pack_state_v1(text,integer)'::regprocedure,
 'public.collection_epoch_capacity_v1(bigint,bigint,text,jsonb)'::regprocedure,
 'public.collection_epoch_liability_v1(bigint)'::regprocedure,
 'public.collection_pool_fundable_v1(integer)'::regprocedure,
 'public.collection_route_backoff_v1(text,text,integer)'::regprocedure,
 'public.collection_index_pending_bill_v1()'::regprocedure,
 'public.collection_note_route_throttle_v1()'::regprocedure,
 'public.collection_set_worker_lanes_v1(integer)'::regprocedure] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  IF f NOT IN ('public.collection_expire_worker_leases_v1(boolean,bigint)'::regprocedure,'public.collection_set_worker_lanes_v1(integer)'::regprocedure,'public.collection_note_route_throttle_v1()'::regprocedure,'public.collection_index_pending_bill_v1()'::regprocedure)
  THEN EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f); END IF;
 END LOOP;
 IF NOT (SELECT had_create FROM pg_temp.governor_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.governor_acl)
 THEN RAISE EXCEPTION '773 schema privileges changed'; END IF;
END $$;
NOTIFY pgrst,'reload schema';
COMMIT;
