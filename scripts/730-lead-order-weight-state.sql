-- Lead-order rewards are paid through the epoch's accepted weight state
-- (docs/gamma-lead-orders.md, "Weights"). These two guards keep the payout
-- ledger and the signed state equal:
--   * a state is refused unless its lead_orders field matches the live payout
--     recorded for its epoch (same hash, and present exactly when that payout
--     paid something);
--   * a live payout is refused once its epoch's state is published.
-- Both take the payout lock, so neither can pass the other half-written.
-- Adds triggers only; changes no existing function, table or row.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE lead_orders730_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $prereq730$
BEGIN
 IF to_regclass('public.lead_order_reward_epochs') IS NULL
  OR to_regclass('public.lab_arena_accepted_weight_states') IS NULL THEN
  RAISE EXCEPTION 'lead order weight guards need lead orders (707) and accepted weight states (202)' USING ERRCODE='55000';
 END IF;
END;
$prereq730$;

CREATE FUNCTION public.lead_order__weight_state_guard_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $state730$
DECLARE paid_hash TEXT; signed JSONB := NEW.state_doc->'lead_orders';
BEGIN
 IF NEW.network<>'finney' OR NEW.netuid<>71 THEN RETURN NEW; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('lead_order_payout',0));
 SELECT e.doc_hash INTO paid_hash FROM public.lead_order_reward_epochs e
  WHERE e.epoch=NEW.epoch AND e.mode='live' AND (e.doc->>'paid_microusd')::NUMERIC>0;
 IF (signed->>'payout_hash') IS DISTINCT FROM paid_hash THEN
  RAISE EXCEPTION 'weight state differs from the lead order payout of epoch %',NEW.epoch USING ERRCODE='23514';
 END IF;
 IF paid_hash IS NOT NULL AND signed->'paid_microusd' IS DISTINCT FROM (
   SELECT jsonb_object_agg(t.miner_hotkey,t.paid) FROM (
    SELECT p->>'miner_hotkey' miner_hotkey,sum((p->>'paid_microusd')::NUMERIC) paid
    FROM public.lead_order_reward_epochs e,jsonb_array_elements(e.doc->'payments') p
    WHERE e.epoch=NEW.epoch GROUP BY 1) t) THEN
  RAISE EXCEPTION 'weight state amounts differ from the lead order payout of epoch %',NEW.epoch USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END;
$state730$;

CREATE FUNCTION public.lead_order__payout_epoch_guard_v1() RETURNS trigger
 LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $payout730$
BEGIN
 -- lead_order_record_payout_v1 already holds the payout lock here.
 IF NEW.mode='live' AND EXISTS(SELECT 1 FROM public.lab_arena_accepted_weight_states s
   WHERE s.network='finney' AND s.netuid=71 AND s.epoch=NEW.epoch) THEN
  RAISE EXCEPTION 'epoch % weights are already signed; lead order rewards stay owed',NEW.epoch USING ERRCODE='55000';
 END IF;
 RETURN NEW;
END;
$payout730$;

CREATE TRIGGER lead_order_weight_state_guard BEFORE INSERT ON public.lab_arena_accepted_weight_states
 FOR EACH ROW EXECUTE FUNCTION public.lead_order__weight_state_guard_v1();
CREATE TRIGGER lead_order_payout_epoch_guard BEFORE INSERT ON public.lead_order_reward_epochs
 FOR EACH ROW EXECUTE FUNCTION public.lead_order__payout_epoch_guard_v1();

DO $functions730$
DECLARE f REGPROCEDURE;
BEGIN
 FOREACH f IN ARRAY ARRAY['public.lead_order__weight_state_guard_v1()'::REGPROCEDURE,
   'public.lead_order__payout_epoch_guard_v1()'::REGPROCEDURE] LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
 END LOOP;
END;
$functions730$;

DO $restore730$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.lead_orders730_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.lead_orders730_acl)THEN
  RAISE EXCEPTION 'lead_orders730 schema ACL differs';END IF;
END;
$restore730$;
COMMIT;
