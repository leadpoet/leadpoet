-- Supplier receivables are USD. Freeze an allocation only in the accepted,
-- signed Finney SN71 state. A target weight is not a receipt of alpha.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE tm771_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
SET LOCAL ROLE lab_arena_owner;

-- The accepted state is the sole journal of scheduled weights. No parallel
-- "paid" event that could commit before the state or survive a failed issue.
CREATE INDEX IF NOT EXISTS tool_market_signed_epochs ON public.lab_arena_accepted_weight_states(epoch)
 WHERE network='finney' AND netuid=71 AND state_doc ? 'tool_market';
CREATE INDEX IF NOT EXISTS tool_market_earning_epochs ON public.tool_market_call_settlements(epoch,call_id)
 WHERE status='settled' AND owed_microusd>0;

CREATE OR REPLACE FUNCTION public.tool_market__scheduled_v1()
RETURNS TABLE (miner TEXT,scheduled NUMERIC) LANGUAGE sql STABLE
SET search_path=pg_catalog,public AS $f$
 SELECT a.key,sum(a.value::numeric)
 FROM public.lab_arena_accepted_weight_states w
 CROSS JOIN LATERAL jsonb_each_text(w.state_doc#>'{tool_market,allocations_microusd}') a
 WHERE w.network='finney' AND w.netuid=71 AND w.state_doc ? 'tool_market'
 GROUP BY a.key
$f$;

CREATE OR REPLACE FUNCTION public.tool_market__outstanding_v1(p_epoch BIGINT)
RETURNS TABLE (miner TEXT,remaining NUMERIC,oldest TIMESTAMPTZ) LANGUAGE sql STABLE
SET search_path=pg_catalog,public AS $f$
 -- Consume accepted targets against the oldest settled calls to that supplier.
 -- Settlement can arrive after its original epoch; callers never advance a cursor past it.
 WITH earnings AS (
  SELECT k.miner_hotkey miner,s.owed_microusd,s.settled_at,
   sum(s.owed_microusd) OVER (PARTITION BY k.miner_hotkey ORDER BY s.settled_at,s.call_id) running
  FROM public.tool_market_call_settlements s JOIN public.tool_market_calls c USING (call_id)
  JOIN public.tool_market_keys k USING (key_id)
  WHERE s.epoch<p_epoch AND s.status='settled' AND s.owed_microusd>0),
 amounts AS (
  SELECT e.miner,e.settled_at,least(e.owed_microusd,e.running-coalesce(p.scheduled,0)) amount
  FROM earnings e LEFT JOIN public.tool_market__scheduled_v1() p USING (miner))
 SELECT a.miner,sum(a.amount),min(a.settled_at) FROM amounts a WHERE a.amount>0 GROUP BY a.miner
$f$;

CREATE OR REPLACE FUNCTION public.tool_market_has_unallocated_v1(p_epoch BIGINT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $f$
 SELECT EXISTS(SELECT FROM public.tool_market__outstanding_v1(p_epoch))
$f$;

CREATE OR REPLACE FUNCTION public.tool_market_epoch_allocations_v1(
 p_epoch BIGINT,p_cap BIGINT,p_emission BIGINT,p_eligible TEXT[])
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $f$
 -- At least one u16 weight quantum; smaller debts remain outstanding. Complete
 -- or partially allocate each supplier in oldest outstanding charge order.
 WITH waiting AS (
  SELECT o.*, sum(remaining) OVER(ORDER BY oldest,miner ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) ahead
  FROM public.tool_market__outstanding_v1(p_epoch) o
  WHERE miner=ANY(p_eligible) AND remaining*65535>=p_emission AND p_emission>0 AND p_cap BETWEEN 1 AND p_emission),
 allocated AS (
  SELECT miner,least(remaining,p_cap-coalesce(ahead,0))::numeric amount FROM waiting)
 SELECT coalesce(jsonb_object_agg(miner,amount::bigint) FILTER(WHERE amount>0 AND amount*65535>=p_emission),'{}')
 FROM allocated
$f$;

-- Scheduled means the accepted signed state included the target. The chain's
-- realized distribution is a separate fact and can also include lab rewards.
CREATE OR REPLACE FUNCTION public.tool_market_emission_statement_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $f$
 WITH accrued AS (
   SELECT coalesce(sum(s.owed_microusd),0) amount
   FROM public.tool_market_keys k JOIN public.tool_market_calls c USING(key_id)
   JOIN public.tool_market_call_settlements s USING(call_id)
   WHERE k.miner_hotkey=p_hotkey AND s.status='settled' AND s.owed_microusd>0),
 scheduled AS (SELECT coalesce(sum(scheduled),0) amount FROM public.tool_market__scheduled_v1() WHERE miner=p_hotkey)
 SELECT jsonb_build_object('accrued_microusd',accrued.amount,'scheduled_microusd',scheduled.amount,
  'unscheduled_microusd',accrued.amount-scheduled.amount,'realized_alpha_rao',NULL) FROM accrued,scheduled
$f$;

CREATE OR REPLACE FUNCTION public.tool_market_guard_epoch_v1() RETURNS trigger LANGUAGE plpgsql
SECURITY DEFINER SET search_path=pg_catalog,public AS $f$
DECLARE overlay jsonb:=NEW.state_doc->'tool_market'; amounts jsonb:=overlay->'allocations_microusd';
 basis jsonb:=NEW.state_doc->'reward_basis'; v jsonb; emission numeric; original_quotes numeric; total numeric;
 redeemable jsonb; prior text;
BEGIN
 IF NEW.network<>'finney' OR NEW.netuid<>71 OR basis->>'schema_version' IS DISTINCT FROM 'leadpoet.collection.fixed_run_quotes.v1'
    OR overlay->>'schema_version' IS DISTINCT FROM 'leadpoet.tool_market.supplier_allocations.v1'
    OR overlay->'epoch' IS DISTINCT FROM to_jsonb(NEW.epoch) OR jsonb_typeof(amounts) IS DISTINCT FROM 'object'
    OR amounts='{}' OR amounts ? (NEW.state_doc->>'burn_hotkey')
    OR NOT EXISTS(SELECT FROM public.tool_market_gamma_source WHERE netuid=71
        AND genesis='0x'||(NEW.state_doc->>'genesis_hash'))
 THEN RAISE EXCEPTION 'invalid tool market payout scope'; END IF;
 IF basis->>'job_count'='0' THEN
   v:=overlay->'valuation';
   IF basis->'valuation' IS DISTINCT FROM 'null'::jsonb THEN RAISE EXCEPTION 'invalid empty lab valuation'; END IF;
 ELSE
   v:=basis->'valuation';
   IF overlay ? 'valuation' THEN RAISE EXCEPTION 'lab valuation must be reused'; END IF;
 END IF;
 emission:=(v->>'emission_microusd')::numeric;
 IF emission IS NULL OR emission<=0 OR emission>9223372036854775807
    OR overlay->'emission_microusd' IS DISTINCT FROM v->'emission_microusd'
    OR v->'block' IS DISTINCT FROM NEW.state_doc->'valid_from_block'
    OR NOT coalesce(((v->>'observed_at')::bigint-(v->>'price_updated_at')::bigint) BETWEEN 0 AND 900,false)
    OR NOT coalesce((v->>'miner_alpha_rao')::numeric>0 AND (v->>'tao_rao_per_alpha')::numeric>0
               AND (v->>'tao_price_microusd')::numeric>0,false)
    OR emission IS DISTINCT FROM div((v->>'miner_alpha_rao')::numeric*(v->>'tao_rao_per_alpha')::numeric*
                                     (v->>'tao_price_microusd')::numeric,1000000000000000000)
 THEN RAISE EXCEPTION 'invalid tool market payout valuation'; END IF;
 SELECT coalesce(sum(value::numeric),0) INTO original_quotes FROM jsonb_each_text(basis->'quotes_microusd');
 SELECT sum(value::numeric) INTO total FROM jsonb_each_text(amounts);
 IF total>emission-original_quotes OR EXISTS(SELECT FROM jsonb_each(amounts) a
    WHERE a.key !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$' OR jsonb_typeof(a.value)<>'number'
      OR a.value::text !~ '^[1-9][0-9]*$' OR a.value::text::numeric*65535<emission) THEN
  RAISE EXCEPTION 'tool market payout exceeds residual or cannot be represented';
 END IF;

 -- Re-read immutable accruals after taking the cross-epoch lock. A repeatable
 -- read transaction could still see the pre-lock debt in a second issuer.
 IF current_setting('transaction_isolation')<>'read committed' THEN
  RAISE EXCEPTION 'tool market target requires read committed issuance';
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('tool_market:emission-issuance',0));
 SELECT state_hash INTO prior FROM public.lab_arena_accepted_weight_states
 WHERE network=NEW.network AND netuid=NEW.netuid AND epoch=NEW.epoch;
 IF FOUND THEN
   IF prior<>NEW.state_hash OR NOT EXISTS(SELECT FROM public.lab_arena_accepted_weight_states
        WHERE state_hash=prior AND state_doc-'signature'=NEW.state_doc-'signature') THEN
     RAISE EXCEPTION 'tool market epoch already accepted';
   END IF;
   RETURN NEW;
 END IF;
 IF EXISTS(SELECT FROM public.lab_arena_accepted_weight_states WHERE network=NEW.network
           AND netuid=NEW.netuid AND epoch>NEW.epoch) THEN RAISE EXCEPTION 'tool market payout epoch is stale'; END IF;
 -- The signer proved eligibility on chain. Independently cap every liability at
 -- its actual settlement balance after all earlier accepted allocations.
 SELECT jsonb_object_agg(miner,remaining) INTO redeemable FROM public.tool_market__outstanding_v1(NEW.epoch);
 IF EXISTS(SELECT FROM jsonb_each_text(amounts) a
           WHERE a.value::numeric>coalesce((redeemable->>a.key)::numeric,0)) THEN
  RAISE EXCEPTION 'tool market payout exceeds supplier receivable';
 END IF;
 RETURN NEW;
END
$f$;

CREATE OR REPLACE TRIGGER tool_market_guard_epoch BEFORE INSERT ON public.lab_arena_accepted_weight_states
FOR EACH ROW WHEN (NEW.state_doc ? 'tool_market') EXECUTE FUNCTION public.tool_market_guard_epoch_v1();

REVOKE ALL ON FUNCTION public.tool_market__scheduled_v1(),public.tool_market__outstanding_v1(bigint),
 public.tool_market_guard_epoch_v1(),public.tool_market_emission_statement_v1(text),public.tool_market_has_unallocated_v1(bigint),
 public.tool_market_epoch_allocations_v1(bigint,bigint,bigint,text[]) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.tool_market_emission_statement_v1(text),public.tool_market_has_unallocated_v1(bigint),public.tool_market_epoch_allocations_v1(bigint,bigint,bigint,text[])
 TO lab_arena_service;
RESET ROLE;
DO $acl$ BEGIN
 IF NOT(SELECT had_create FROM pg_temp.tm771_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.tm771_acl)
 THEN RAISE EXCEPTION 'tool market 771 schema ACL differs'; END IF;
END $acl$;
NOTIFY pgrst,'reload schema';
COMMIT;
