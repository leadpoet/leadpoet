-- Gamma lead orders (docs/gamma-lead-orders.md). Off by default: nothing
-- calls these functions unless LEADPOET_LEAD_ORDERS_MODE is staff, shadow or
-- live. Every table is append-only (no UPDATE, DELETE or TRUNCATE); state
-- moves forward by inserting claim, settlement, finish and payment rows.
-- RLS is on with no policy and no table grant: the Arena service role reaches
-- the rows only through the SECURITY DEFINER functions below, as the
-- neighbouring lab_arena tables do. Touches no Arena table or function.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE lead_orders703_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $prereq703$
BEGIN
 IF to_regprocedure('public.lab_arena_append_only_v1()') IS NULL
  OR to_regrole('lab_arena_owner') IS NULL OR to_regrole('lab_arena_service') IS NULL THEN
  RAISE EXCEPTION 'lead orders need the Lab Arena roles and append-only trigger (179)' USING ERRCODE='55000';
 END IF;
END;
$prereq703$;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.lead_order_policies (
  version INTEGER PRIMARY KEY CHECK(version>0),
  policy_hash TEXT NOT NULL UNIQUE CHECK(policy_hash ~ '^sha256:[0-9a-f]{64}$'),
  policy_doc JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_gamma_readings (
  reading_id BIGSERIAL PRIMARY KEY,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  block_number BIGINT NOT NULL CHECK(block_number>=0),
  block_hash TEXT NOT NULL CHECK(block_hash ~ '^0x[0-9a-f]{64}$'),
  spent_units NUMERIC(39,0) NOT NULL CHECK(spent_units>=0),
  available BOOLEAN NOT NULL,
  read_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS lead_order_gamma_readings_hotkey_idx
 ON public.lead_order_gamma_readings(hotkey,block_number DESC,reading_id DESC);

CREATE TABLE IF NOT EXISTS public.lead_order_buyers (
  registration_id UUID PRIMARY KEY,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  email TEXT NOT NULL CHECK(length(email) BETWEEN 3 AND 254),
  company TEXT NOT NULL CHECK(length(company) BETWEEN 1 AND 200),
  country TEXT NOT NULL CHECK(country ~ '^[A-Z]{2}$'),
  terms_version TEXT NOT NULL CHECK(terms_version ~ '^[A-Za-z0-9._-]{1,32}$'),
  token_hash TEXT NOT NULL UNIQUE CHECK(token_hash ~ '^sha256:[0-9a-f]{64}$'),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS lead_order_buyers_hotkey_idx ON public.lead_order_buyers(hotkey,created_at DESC);

CREATE TABLE IF NOT EXISTS public.lead_order_buyer_confirmations (
  token_hash TEXT PRIMARY KEY REFERENCES public.lead_order_buyers(token_hash),
  confirmed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_nonces (
  hotkey TEXT NOT NULL,
  signature_hash TEXT NOT NULL CHECK(signature_hash ~ '^sha256:[0-9a-f]{64}$'),
  signed_at BIGINT NOT NULL,
  used_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(hotkey,signature_hash)
);

CREATE TABLE IF NOT EXISTS public.lead_orders (
  order_id UUID PRIMARY KEY,
  sequence BIGSERIAL NOT NULL UNIQUE,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  mode TEXT NOT NULL CHECK(mode IN ('staff','shadow','live')),
  policy_version INTEGER NOT NULL REFERENCES public.lead_order_policies(version),
  leads_requested INTEGER NOT NULL CHECK(leads_requested>0),
  hold_microusd BIGINT NOT NULL CHECK(hold_microusd>=0),
  spend_ceiling_microusd BIGINT NOT NULL CHECK(spend_ceiling_microusd>0),
  price_per_lead_microusd BIGINT NOT NULL CHECK(price_per_lead_microusd>0),
  min_charge_ratio_ppm INTEGER NOT NULL CHECK(min_charge_ratio_ppm BETWEEN 1 AND 1000000),
  deadline_minutes INTEGER NOT NULL CHECK(deadline_minutes BETWEEN 1 AND 1440),
  request_doc JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS lead_orders_hotkey_idx ON public.lead_orders(hotkey,created_at DESC);

CREATE TABLE IF NOT EXISTS public.lead_order_claims (
  order_id UUID PRIMARY KEY REFERENCES public.lead_orders(order_id),
  token_hash TEXT NOT NULL UNIQUE CHECK(token_hash ~ '^sha256:[0-9a-f]{64}$'),
  claimed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  deadline TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS public.lead_order_finishes (
  order_id UUID PRIMARY KEY REFERENCES public.lead_order_claims(order_id),
  sequence BIGSERIAL NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK(status IN ('delivered','failed')),
  leads_delivered INTEGER NOT NULL CHECK(leads_delivered>=0),
  delivery_url TEXT CHECK(delivery_url IS NULL OR delivery_url ~ '^https://'),
  failure_reason TEXT CHECK(failure_reason IS NULL OR length(failure_reason)<=200),
  spend_microusd BIGINT NOT NULL CHECK(spend_microusd>=0),
  charge_microusd BIGINT NOT NULL CHECK(charge_microusd>=0),
  settled_epoch BIGINT NOT NULL CHECK(settled_epoch>=0),
  finished_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK((status='delivered')=(delivery_url IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS public.lead_order_keys (
  key_id TEXT PRIMARY KEY CHECK(key_id ~ '^[a-z0-9-]{1,64}$' AND key_id NOT LIKE 'leadpoet-%'),
  sequence BIGSERIAL NOT NULL UNIQUE,
  provider TEXT NOT NULL CHECK(provider IN ('openrouter','deepline')),
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  submission_id TEXT NOT NULL UNIQUE CHECK(submission_id='lead-order:'||key_id),
  ciphertext_b64 TEXT NOT NULL CHECK(length(ciphertext_b64) BETWEEN 1 AND 16384),
  internal_fingerprint TEXT NOT NULL UNIQUE CHECK(internal_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
  public_fingerprint TEXT NOT NULL CHECK(public_fingerprint ~ '^sha256:[0-9a-f]{32}$'),
  registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_key_revocations (
  key_id TEXT PRIMARY KEY REFERENCES public.lead_order_keys(key_id),
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 200),
  revoked_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_key_exhaustions (
  order_id UUID NOT NULL REFERENCES public.lead_order_claims(order_id),
  key_id TEXT NOT NULL,
  reason TEXT NOT NULL CHECK(reason IN ('unauthorized','payment_required','forbidden','rate_limited_no_retry','spent_limit')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(order_id,key_id)
);

CREATE TABLE IF NOT EXISTS public.lead_order_calls (
  call_id UUID PRIMARY KEY,
  sequence BIGSERIAL NOT NULL UNIQUE,
  order_id UUID NOT NULL REFERENCES public.lead_order_claims(order_id),
  provider TEXT NOT NULL CHECK(provider IN ('openrouter','deepline')),
  key_id TEXT NOT NULL CHECK(key_id ~ '^[a-z0-9-]{1,64}$'),
  miner_hotkey TEXT CHECK(miner_hotkey IS NULL OR miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  public_fingerprint TEXT NOT NULL CHECK(public_fingerprint ~ '^sha256:[0-9a-f]{32}$'),
  reserved_microusd BIGINT NOT NULL CHECK(reserved_microusd>0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK((miner_hotkey IS NULL)=(key_id LIKE 'leadpoet-%'))
);
CREATE INDEX IF NOT EXISTS lead_order_calls_order_idx ON public.lead_order_calls(order_id,sequence);

CREATE TABLE IF NOT EXISTS public.lead_order_call_settlements (
  call_id UUID PRIMARY KEY REFERENCES public.lead_order_calls(call_id),
  status TEXT NOT NULL CHECK(status IN ('settled','uncertain')),
  cost_microusd BIGINT CHECK(cost_microusd IS NULL OR cost_microusd>=0),
  http_status INTEGER CHECK(http_status IS NULL OR http_status BETWEEN 100 AND 599),
  settled_epoch BIGINT NOT NULL CHECK(settled_epoch>=0),
  settled_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK((status='settled')=(cost_microusd IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS public.lead_order_rewards_owed (
  order_id UUID NOT NULL REFERENCES public.lead_order_finishes(order_id),
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  owed_microusd BIGINT NOT NULL CHECK(owed_microusd>0),
  PRIMARY KEY(order_id,miner_hotkey)
);

CREATE TABLE IF NOT EXISTS public.lead_order_receipts (
  order_id UUID PRIMARY KEY REFERENCES public.lead_order_finishes(order_id),
  receipt_hash TEXT NOT NULL UNIQUE CHECK(receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
  receipt_doc JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_reward_epochs (
  epoch BIGINT PRIMARY KEY CHECK(epoch>=0),
  mode TEXT NOT NULL CHECK(mode IN ('shadow','live')),
  doc_hash TEXT NOT NULL UNIQUE CHECK(doc_hash ~ '^sha256:[0-9a-f]{64}$'),
  doc JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_reward_payments (
  epoch BIGINT NOT NULL REFERENCES public.lead_order_reward_epochs(epoch),
  order_id UUID NOT NULL,
  miner_hotkey TEXT NOT NULL,
  paid_microusd BIGINT NOT NULL CHECK(paid_microusd>0),
  PRIMARY KEY(epoch,order_id,miner_hotkey),
  FOREIGN KEY(order_id,miner_hotkey) REFERENCES public.lead_order_rewards_owed(order_id,miner_hotkey)
);

-- Ownership, RLS, no table grants, append-only triggers (row and TRUNCATE).
DO $tables703$
DECLARE t TEXT;
BEGIN
 FOREACH t IN ARRAY ARRAY['lead_order_policies','lead_order_gamma_readings','lead_order_buyers',
   'lead_order_buyer_confirmations','lead_order_nonces','lead_orders','lead_order_claims',
   'lead_order_finishes','lead_order_keys','lead_order_key_revocations','lead_order_key_exhaustions',
   'lead_order_calls','lead_order_call_settlements','lead_order_rewards_owed','lead_order_receipts',
   'lead_order_reward_epochs','lead_order_reward_payments'] LOOP
  EXECUTE format('ALTER TABLE public.%I OWNER TO lab_arena_owner',t);
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',t);
  EXECUTE format('DROP TRIGGER IF EXISTS %I ON public.%I',t||'_append_only',t);
  EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1()',t||'_append_only',t);
  EXECUTE format('DROP TRIGGER IF EXISTS %I ON public.%I',t||'_no_truncate',t);
  EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON public.%I FOR EACH STATEMENT EXECUTE FUNCTION public.lab_arena_append_only_v1()',t||'_no_truncate',t);
 END LOOP;
END;
$tables703$;
ALTER SEQUENCE public.lead_order_gamma_readings_reading_id_seq OWNER TO lab_arena_owner;

-- ---------------------------------------------------------------------------
-- Internal readers
-- ---------------------------------------------------------------------------

-- Committed spend: settled cost, or the reservation while a call is open or
-- its cost is uncertain. This is what the spend ceiling is checked against.
CREATE OR REPLACE FUNCTION public.lead_order__committed_microusd_v1(p_order_id UUID)
RETURNS BIGINT LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $committed703$
 SELECT coalesce(sum(CASE WHEN s.status='settled' THEN s.cost_microusd ELSE c.reserved_microusd END),0)::BIGINT
 FROM public.lead_order_calls c LEFT JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id
 WHERE c.order_id=p_order_id;
$committed703$;

-- Charged spend: settled cost only. Uncertain calls are never charged or rewarded.
CREATE OR REPLACE FUNCTION public.lead_order__settled_microusd_v1(p_order_id UUID)
RETURNS BIGINT LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $settled703$
 SELECT coalesce(sum(s.cost_microusd),0)::BIGINT
 FROM public.lead_order_calls c JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id
 WHERE c.order_id=p_order_id AND s.status='settled';
$settled703$;

CREATE OR REPLACE FUNCTION public.lead_order_balance_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $balance703$
DECLARE r public.lead_order_gamma_readings; charges BIGINT; holds BIGINT; units NUMERIC; credit BIGINT;
BEGIN
 SELECT * INTO r FROM public.lead_order_gamma_readings WHERE hotkey=p_hotkey
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 SELECT coalesce(sum(f.charge_microusd),0) INTO charges FROM public.lead_orders o
  JOIN public.lead_order_finishes f ON f.order_id=o.order_id WHERE o.hotkey=p_hotkey AND o.mode='live';
 SELECT coalesce(sum(o.hold_microusd),0) INTO holds FROM public.lead_orders o
  WHERE o.hotkey=p_hotkey AND o.mode='live'
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_finishes f WHERE f.order_id=o.order_id);
 units:=CASE WHEN r.available THEN r.spent_units ELSE 0 END;
 -- 10^9 Gamma units per USD: 1000 units per micro-USD, rounded down.
 credit:=floor(coalesce(units,0)/1000)::BIGINT;
 RETURN jsonb_build_object('available',coalesce(r.available,FALSE),'gamma_units',coalesce(units,0)::TEXT,
  'block_number',r.block_number,'charges_microusd',charges,'holds_microusd',holds,
  'balance_microusd',credit-charges-holds);
END;
$balance703$;

-- ---------------------------------------------------------------------------
-- Writers
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.lead_order_record_policy_v1(p_version INTEGER,p_hash TEXT,p_doc JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $policy703$
DECLARE existing TEXT;
BEGIN
 SELECT policy_hash INTO existing FROM public.lead_order_policies WHERE version=p_version;
 IF existing IS NOT NULL THEN
  IF existing<>p_hash THEN RAISE EXCEPTION 'lead order policy version % already has another hash',p_version USING ERRCODE='23505'; END IF;
  RETURN jsonb_build_object('status','exists');
 END IF;
 IF p_doc->>'policy_hash' IS DISTINCT FROM p_hash OR (p_doc->>'version')::INTEGER IS DISTINCT FROM p_version THEN
  RAISE EXCEPTION 'lead order policy document differs from its hash or version' USING ERRCODE='22023';
 END IF;
 INSERT INTO public.lead_order_policies(version,policy_hash,policy_doc) VALUES(p_version,p_hash,p_doc);
 RETURN jsonb_build_object('status','recorded');
END;
$policy703$;

CREATE OR REPLACE FUNCTION public.lead_order_record_gamma_reading_v1(
 p_hotkey TEXT,p_block_number BIGINT,p_block_hash TEXT,p_units NUMERIC,p_available BOOLEAN)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reading703$
DECLARE prior public.lead_order_gamma_readings;
BEGIN
 SELECT * INTO prior FROM public.lead_order_gamma_readings WHERE hotkey=p_hotkey AND available
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 -- The counter only grows. A smaller finalized value means a different
 -- chain or a broken read: refuse rather than shrink a balance silently.
 IF p_available AND prior.reading_id IS NOT NULL AND p_block_number>=prior.block_number
  AND p_units<prior.spent_units THEN
  RAISE EXCEPTION 'GammaSpent went backwards' USING ERRCODE='22023';
 END IF;
 INSERT INTO public.lead_order_gamma_readings(hotkey,block_number,block_hash,spent_units,available)
 VALUES(p_hotkey,p_block_number,lower(p_block_hash),CASE WHEN p_available THEN p_units ELSE 0 END,p_available);
 RETURN public.lead_order_balance_v1(p_hotkey);
END;
$reading703$;

CREATE OR REPLACE FUNCTION public.lead_order_register_buyer_v1(
 p_registration_id UUID,p_hotkey TEXT,p_email TEXT,p_company TEXT,p_country TEXT,p_terms TEXT,p_token_hash TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $register703$
BEGIN
 INSERT INTO public.lead_order_buyers(registration_id,hotkey,email,company,country,terms_version,token_hash)
 VALUES(p_registration_id,p_hotkey,lower(p_email),p_company,p_country,p_terms,p_token_hash);
 RETURN jsonb_build_object('status','pending','registration_id',p_registration_id);
END;
$register703$;

CREATE OR REPLACE FUNCTION public.lead_order_confirm_buyer_v1(p_token_hash TEXT,p_max_age_seconds INTEGER)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $confirm703$
DECLARE b public.lead_order_buyers;
BEGIN
 SELECT * INTO b FROM public.lead_order_buyers WHERE token_hash=p_token_hash;
 IF b.registration_id IS NULL THEN RETURN jsonb_build_object('status','unknown'); END IF;
 IF EXISTS(SELECT 1 FROM public.lead_order_buyer_confirmations WHERE token_hash=p_token_hash) THEN
  RETURN jsonb_build_object('status','confirmed','hotkey',b.hotkey);
 END IF;
 IF b.created_at < clock_timestamp()-make_interval(secs=>p_max_age_seconds) THEN
  RETURN jsonb_build_object('status','expired');
 END IF;
 INSERT INTO public.lead_order_buyer_confirmations(token_hash) VALUES(p_token_hash);
 RETURN jsonb_build_object('status','confirmed','hotkey',b.hotkey);
END;
$confirm703$;

-- The buyer is the hotkey's latest registration; it counts once confirmed.
CREATE OR REPLACE FUNCTION public.lead_order_buyer_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $buyer703$
 SELECT coalesce((SELECT jsonb_build_object('registered',TRUE,'terms_version',b.terms_version,'country',b.country,
   'confirmed',EXISTS(SELECT 1 FROM public.lead_order_buyer_confirmations c WHERE c.token_hash=b.token_hash))
  FROM public.lead_order_buyers b WHERE b.hotkey=p_hotkey ORDER BY b.created_at DESC,b.registration_id DESC LIMIT 1),
  jsonb_build_object('registered',FALSE,'confirmed',FALSE));
$buyer703$;

CREATE OR REPLACE FUNCTION public.lead_order_use_nonce_v1(p_hotkey TEXT,p_signature_hash TEXT,p_signed_at BIGINT)
RETURNS BOOLEAN LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $nonce703$
BEGIN
 INSERT INTO public.lead_order_nonces(hotkey,signature_hash,signed_at) VALUES(p_hotkey,p_signature_hash,p_signed_at)
 ON CONFLICT DO NOTHING;
 RETURN FOUND;
END;
$nonce703$;

-- Place an order atomically per hotkey: minimum, rolling 24 h cap, and (live
-- only) Gamma balance against the hold. Money numbers come from the policy
-- version the caller names, so a receipt can always be recomputed.
CREATE OR REPLACE FUNCTION public.lead_order_place_v1(
 p_order_id UUID,p_hotkey TEXT,p_mode TEXT,p_policy_version INTEGER,p_leads INTEGER,p_request_doc JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $place703$
DECLARE pol JSONB; price BIGINT; ceiling BIGINT; ratio INTEGER; min_leads INTEGER; daily INTEGER;
 deadline_minutes INTEGER; today BIGINT; bal JSONB; hold BIGINT;
BEGIN
 SELECT policy_doc INTO pol FROM public.lead_order_policies WHERE version=p_policy_version;
 IF pol IS NULL THEN RAISE EXCEPTION 'unknown lead order policy %',p_policy_version USING ERRCODE='22023'; END IF;
 price:=round((pol->>'price_per_lead_usd')::NUMERIC*1000000);
 ceiling:=round((pol->>'spend_ceiling_per_lead_usd')::NUMERIC*1000000);
 ratio:=round((pol->>'min_charge_ratio')::NUMERIC*1000000);
 min_leads:=(pol->>'min_leads_per_order')::INTEGER;
 daily:=(pol->>'daily_leads_per_hotkey')::INTEGER;
 deadline_minutes:=(pol->>'order_deadline_minutes')::INTEGER;
 PERFORM pg_advisory_xact_lock(hashtextextended('lead_order_hotkey:'||p_hotkey,0));
 IF EXISTS(SELECT 1 FROM public.lead_orders WHERE order_id=p_order_id) THEN
  RETURN jsonb_build_object('status','exists');
 END IF;
 hold:=p_leads::BIGINT*price;
 IF p_leads<min_leads THEN RETURN jsonb_build_object('status','below_minimum'); END IF;
 SELECT coalesce(sum(leads_requested),0) INTO today FROM public.lead_orders
  WHERE hotkey=p_hotkey AND created_at>clock_timestamp()-interval '24 hours';
 IF today+p_leads>daily THEN RETURN jsonb_build_object('status','daily_cap'); END IF;
 IF p_mode='live' THEN
  bal:=public.lead_order_balance_v1(p_hotkey);
  IF NOT (bal->>'available')::BOOLEAN THEN RETURN jsonb_build_object('status','gamma_unavailable'); END IF;
  IF (bal->>'balance_microusd')::BIGINT<hold THEN
   RETURN jsonb_build_object('status','insufficient_balance','balance_microusd',(bal->>'balance_microusd')::BIGINT);
  END IF;
 END IF;
 INSERT INTO public.lead_orders(order_id,hotkey,mode,policy_version,leads_requested,hold_microusd,
  spend_ceiling_microusd,price_per_lead_microusd,min_charge_ratio_ppm,deadline_minutes,request_doc)
 VALUES(p_order_id,p_hotkey,p_mode,p_policy_version,p_leads,hold,p_leads::BIGINT*ceiling,price,ratio,
  deadline_minutes,p_request_doc);
 RETURN jsonb_build_object('status','placed','order_id',p_order_id,'hold_microusd',hold,
  'spend_ceiling_microusd',p_leads::BIGINT*ceiling);
END;
$place703$;

-- Claim the oldest queued order and bind the one-order token to it.
CREATE OR REPLACE FUNCTION public.lead_order_claim_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $claim703$
DECLARE o public.lead_orders; d TIMESTAMPTZ;
BEGIN
 SELECT * INTO o FROM public.lead_orders q
  WHERE NOT EXISTS(SELECT 1 FROM public.lead_order_claims c WHERE c.order_id=q.order_id)
  ORDER BY q.sequence LIMIT 1 FOR UPDATE SKIP LOCKED;
 IF o.order_id IS NULL THEN RETURN NULL; END IF;
 d:=clock_timestamp()+make_interval(mins=>o.deadline_minutes);
 INSERT INTO public.lead_order_claims(order_id,token_hash,deadline) VALUES(o.order_id,p_token_hash,d);
 RETURN jsonb_build_object('order_id',o.order_id,'hotkey',o.hotkey,'mode',o.mode,
  'leads_requested',o.leads_requested,'deadline',to_char(d AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'),
  'spend_ceiling_microusd',o.spend_ceiling_microusd,'request_doc',o.request_doc);
END;
$claim703$;

-- What the broker and heartbeat need for a token: whether calls may run.
CREATE OR REPLACE FUNCTION public.lead_order_by_token_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $token703$
 SELECT jsonb_build_object('order_id',o.order_id,'mode',o.mode,'spend_ceiling_microusd',o.spend_ceiling_microusd,
  'committed_microusd',public.lead_order__committed_microusd_v1(o.order_id),
  'finished',EXISTS(SELECT 1 FROM public.lead_order_finishes f WHERE f.order_id=o.order_id),
  'expired',c.deadline<=clock_timestamp())
 FROM public.lead_order_claims c JOIN public.lead_orders o ON o.order_id=c.order_id
 WHERE c.token_hash=p_token_hash;
$token703$;

-- Reserve a call's worst-case cost before it is sent. Refused at the ceiling,
-- after the deadline, and once the order is finished.
CREATE OR REPLACE FUNCTION public.lead_order_reserve_call_v1(
 p_call_id UUID,p_order_id UUID,p_provider TEXT,p_key_id TEXT,p_miner_hotkey TEXT,p_public_fingerprint TEXT,p_reserve BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reserve703$
DECLARE o public.lead_orders; c public.lead_order_claims; committed BIGINT;
BEGIN
 SELECT * INTO o FROM public.lead_orders WHERE order_id=p_order_id FOR UPDATE;
 SELECT * INTO c FROM public.lead_order_claims WHERE order_id=p_order_id;
 IF o.order_id IS NULL OR c.order_id IS NULL THEN RETURN jsonb_build_object('status','unknown'); END IF;
 IF EXISTS(SELECT 1 FROM public.lead_order_finishes WHERE order_id=p_order_id) THEN
  RETURN jsonb_build_object('status','closed');
 END IF;
 IF c.deadline<=clock_timestamp() THEN RETURN jsonb_build_object('status','expired'); END IF;
 IF EXISTS(SELECT 1 FROM public.lead_order_key_exhaustions WHERE order_id=p_order_id AND key_id=p_key_id) THEN
  RETURN jsonb_build_object('status','key_exhausted');
 END IF;
 committed:=public.lead_order__committed_microusd_v1(p_order_id);
 IF committed+p_reserve>o.spend_ceiling_microusd THEN
  RETURN jsonb_build_object('status','ceiling','committed_microusd',committed);
 END IF;
 INSERT INTO public.lead_order_calls(call_id,order_id,provider,key_id,miner_hotkey,public_fingerprint,reserved_microusd)
 VALUES(p_call_id,p_order_id,p_provider,p_key_id,p_miner_hotkey,p_public_fingerprint,p_reserve);
 RETURN jsonb_build_object('status','reserved','committed_microusd',committed+p_reserve);
END;
$reserve703$;

CREATE OR REPLACE FUNCTION public.lead_order_settle_call_v1(
 p_call_id UUID,p_status TEXT,p_cost BIGINT,p_http_status INTEGER,p_epoch BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $settle703$
BEGIN
 INSERT INTO public.lead_order_call_settlements(call_id,status,cost_microusd,http_status,settled_epoch)
 VALUES(p_call_id,p_status,CASE WHEN p_status='settled' THEN p_cost END,p_http_status,p_epoch)
 ON CONFLICT(call_id) DO NOTHING;
 RETURN jsonb_build_object('status',CASE WHEN FOUND THEN 'settled' ELSE 'exists' END);
END;
$settle703$;

CREATE OR REPLACE FUNCTION public.lead_order_exhaust_key_v1(p_order_id UUID,p_key_id TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $exhaust703$
BEGIN
 INSERT INTO public.lead_order_key_exhaustions(order_id,key_id,reason) VALUES(p_order_id,p_key_id,p_reason)
 ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status','recorded');
END;
$exhaust703$;

CREATE OR REPLACE FUNCTION public.lead_order_exhausted_keys_v1(p_order_id UUID)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $exhausted703$
 SELECT coalesce(jsonb_agg(key_id ORDER BY key_id),'[]'::JSONB) FROM public.lead_order_key_exhaustions WHERE order_id=p_order_id;
$exhausted703$;

-- Finish once. charge = min(hold, max(delivered x price, ceil(ratio x settled spend))).
CREATE OR REPLACE FUNCTION public.lead_order_finish_v1(
 p_order_id UUID,p_status TEXT,p_delivered INTEGER,p_delivery_url TEXT,p_failure_reason TEXT,p_epoch BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $finish703$
DECLARE o public.lead_orders; f public.lead_order_finishes; spend BIGINT; charge BIGINT; delivered INTEGER;
BEGIN
 SELECT * INTO o FROM public.lead_orders WHERE order_id=p_order_id FOR UPDATE;
 IF o.order_id IS NULL OR NOT EXISTS(SELECT 1 FROM public.lead_order_claims WHERE order_id=p_order_id) THEN
  RETURN jsonb_build_object('status','unknown');
 END IF;
 SELECT * INTO f FROM public.lead_order_finishes WHERE order_id=p_order_id;
 IF f.order_id IS NOT NULL THEN
  RETURN jsonb_build_object('status','exists','order_status',f.status,'charge_microusd',f.charge_microusd);
 END IF;
 IF p_delivered<0 OR p_delivered>o.leads_requested THEN
  RAISE EXCEPTION 'leads_delivered is outside the order' USING ERRCODE='22023';
 END IF;
 -- A failed order delivers nothing and pays the minimum charge on spend so far.
 delivered:=CASE WHEN p_status='delivered' THEN p_delivered ELSE 0 END;
 spend:=public.lead_order__settled_microusd_v1(p_order_id);
 charge:=LEAST(o.hold_microusd,GREATEST(delivered::BIGINT*o.price_per_lead_microusd,
  div(spend::NUMERIC*o.min_charge_ratio_ppm+999999,1000000)::BIGINT));
 INSERT INTO public.lead_order_finishes(order_id,status,leads_delivered,delivery_url,failure_reason,
  spend_microusd,charge_microusd,settled_epoch)
 VALUES(p_order_id,p_status,delivered,CASE WHEN p_status='delivered' THEN p_delivery_url END,
  CASE WHEN p_status='failed' THEN left(coalesce(p_failure_reason,'failed'),200) END,spend,charge,p_epoch);
 RETURN jsonb_build_object('status','finished','order_status',p_status,'charge_microusd',charge,'spend_microusd',spend);
END;
$finish703$;

CREATE OR REPLACE FUNCTION public.lead_order_overdue_v1(p_grace_seconds INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $overdue703$
 SELECT coalesce(jsonb_agg(c.order_id ORDER BY c.deadline),'[]'::JSONB) FROM public.lead_order_claims c
 WHERE c.deadline+make_interval(secs=>p_grace_seconds)<=clock_timestamp()
  AND NOT EXISTS(SELECT 1 FROM public.lead_order_finishes f WHERE f.order_id=c.order_id);
$overdue703$;

CREATE OR REPLACE FUNCTION public.lead_order_view_v1(p_order_id UUID)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $view703$
 SELECT jsonb_build_object('order_id',o.order_id,'hotkey',o.hotkey,'mode',o.mode,'policy_version',o.policy_version,
  'leads_requested',o.leads_requested,'hold_microusd',o.hold_microusd,'spend_ceiling_microusd',o.spend_ceiling_microusd,
  'status',CASE WHEN f.order_id IS NOT NULL THEN f.status WHEN c.order_id IS NOT NULL THEN 'running' ELSE 'queued' END,
  'leads_delivered',f.leads_delivered,'charge_microusd',f.charge_microusd,'spend_microusd',f.spend_microusd,
  'settled_epoch',f.settled_epoch,'delivery_url',f.delivery_url,'failure_reason',f.failure_reason,
  'deadline',to_char(c.deadline AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'),
  'expired',coalesce(c.deadline<=clock_timestamp(),FALSE),
  'committed_microusd',public.lead_order__committed_microusd_v1(o.order_id),
  'receipt',r.receipt_doc)
 FROM public.lead_orders o LEFT JOIN public.lead_order_claims c ON c.order_id=o.order_id
 LEFT JOIN public.lead_order_finishes f ON f.order_id=o.order_id
 LEFT JOIN public.lead_order_receipts r ON r.order_id=o.order_id
 WHERE o.order_id=p_order_id;
$view703$;

CREATE OR REPLACE FUNCTION public.lead_order_calls_v1(p_order_id UUID)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $calls703$
 SELECT coalesce(jsonb_agg(jsonb_build_object('call_id',c.call_id,'sequence',c.sequence,'provider',c.provider,
   'key_id',c.key_id,'miner_hotkey',c.miner_hotkey,'public_fingerprint',c.public_fingerprint,
   'reserved_microusd',c.reserved_microusd,'status',coalesce(s.status,'open'),'cost_microusd',s.cost_microusd)
  ORDER BY c.sequence),'[]'::JSONB)
 FROM public.lead_order_calls c LEFT JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id
 WHERE c.order_id=p_order_id;
$calls703$;

CREATE OR REPLACE FUNCTION public.lead_order_record_rewards_v1(p_order_id UUID,p_rewards JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $rewards703$
DECLARE k TEXT; v JSONB;
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.lead_order_finishes WHERE order_id=p_order_id) THEN
  RAISE EXCEPTION 'order is not finished' USING ERRCODE='55000';
 END IF;
 IF EXISTS(SELECT 1 FROM public.lead_order_rewards_owed WHERE order_id=p_order_id) THEN
  IF (SELECT jsonb_object_agg(miner_hotkey,owed_microusd) FROM public.lead_order_rewards_owed WHERE order_id=p_order_id)
     IS DISTINCT FROM p_rewards THEN
   RAISE EXCEPTION 'order rewards were already recorded differently' USING ERRCODE='23505';
  END IF;
  RETURN jsonb_build_object('status','exists');
 END IF;
 FOR k,v IN SELECT * FROM jsonb_each(p_rewards) LOOP
  INSERT INTO public.lead_order_rewards_owed(order_id,miner_hotkey,owed_microusd) VALUES(p_order_id,k,(v#>>'{}')::BIGINT);
 END LOOP;
 RETURN jsonb_build_object('status','recorded');
END;
$rewards703$;

CREATE OR REPLACE FUNCTION public.lead_order_record_receipt_v1(p_order_id UUID,p_hash TEXT,p_doc JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $receipt703$
DECLARE existing TEXT;
BEGIN
 SELECT receipt_hash INTO existing FROM public.lead_order_receipts WHERE order_id=p_order_id;
 IF existing IS NOT NULL THEN
  IF existing<>p_hash THEN RAISE EXCEPTION 'order already has another receipt' USING ERRCODE='23505'; END IF;
  RETURN jsonb_build_object('status','exists');
 END IF;
 INSERT INTO public.lead_order_receipts(order_id,receipt_hash,receipt_doc) VALUES(p_order_id,p_hash,p_doc);
 RETURN jsonb_build_object('status','recorded');
END;
$receipt703$;

CREATE OR REPLACE FUNCTION public.lead_order_register_key_v1(
 p_key_id TEXT,p_provider TEXT,p_miner_hotkey TEXT,p_submission_id TEXT,p_ciphertext TEXT,p_internal_fp TEXT,p_public_fp TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $key703$
BEGIN
 -- One active key per miner and provider: a new key needs the old one revoked.
 IF EXISTS(SELECT 1 FROM public.lead_order_keys k WHERE k.miner_hotkey=p_miner_hotkey AND k.provider=p_provider
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_key_revocations r WHERE r.key_id=k.key_id)) THEN
  RETURN jsonb_build_object('status','active_key_exists');
 END IF;
 INSERT INTO public.lead_order_keys(key_id,provider,miner_hotkey,submission_id,ciphertext_b64,internal_fingerprint,public_fingerprint)
 VALUES(p_key_id,p_provider,p_miner_hotkey,p_submission_id,p_ciphertext,p_internal_fp,p_public_fp);
 RETURN jsonb_build_object('status','registered','key_id',p_key_id);
END;
$key703$;

CREATE OR REPLACE FUNCTION public.lead_order_revoke_key_v1(p_key_id TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $revoke703$
BEGIN
 INSERT INTO public.lead_order_key_revocations(key_id,reason) VALUES(p_key_id,p_reason) ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status','revoked');
END;
$revoke703$;

CREATE OR REPLACE FUNCTION public.lead_order_keys_v1(p_provider TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $keys703$
 SELECT coalesce(jsonb_agg(jsonb_build_object('key_id',k.key_id,'provider',k.provider,'miner_hotkey',k.miner_hotkey,
   'sequence',k.sequence,'ciphertext_b64',k.ciphertext_b64,'internal_fingerprint',k.internal_fingerprint) ORDER BY k.sequence),'[]'::JSONB)
 FROM public.lead_order_keys k
 WHERE k.provider=p_provider AND NOT EXISTS(SELECT 1 FROM public.lead_order_key_revocations r WHERE r.key_id=k.key_id);
$keys703$;

-- Settled spend per key over the ranking window, from the call ledger.
CREATE OR REPLACE FUNCTION public.lead_order_key_spend_v1(p_since_epoch BIGINT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $spend703$
 SELECT coalesce(jsonb_object_agg(key_id,total),'{}'::JSONB) FROM (
  SELECT c.key_id,sum(s.cost_microusd)::BIGINT total FROM public.lead_order_calls c
  JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id
  WHERE s.status='settled' AND s.settled_epoch>=p_since_epoch GROUP BY c.key_id) t;
$spend703$;

CREATE OR REPLACE FUNCTION public.lead_order_owed_v1(p_through_epoch BIGINT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $owed703$
 SELECT coalesce(jsonb_agg(jsonb_build_object('order_id',w.order_id,'miner_hotkey',w.miner_hotkey,
   'settled_epoch',f.settled_epoch,'settled_sequence',f.sequence,'owed_microusd',w.owed_microusd,
   'paid_microusd',coalesce((SELECT sum(p.paid_microusd) FROM public.lead_order_reward_payments p
     WHERE p.order_id=w.order_id AND p.miner_hotkey=w.miner_hotkey),0)) ORDER BY f.settled_epoch,f.sequence,w.miner_hotkey),'[]'::JSONB)
 FROM public.lead_order_rewards_owed w JOIN public.lead_order_finishes f ON f.order_id=w.order_id
 WHERE f.settled_epoch<=p_through_epoch;
$owed703$;

-- Record one epoch's payout document and payments. Never pays an entry past
-- what it is owed; an epoch is recorded once.
CREATE OR REPLACE FUNCTION public.lead_order_record_payout_v1(p_epoch BIGINT,p_mode TEXT,p_doc_hash TEXT,p_doc JSONB,p_payments JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $payout703$
DECLARE existing TEXT; p JSONB;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lead_order_payout',0));
 SELECT doc_hash INTO existing FROM public.lead_order_reward_epochs WHERE epoch=p_epoch;
 IF existing IS NOT NULL THEN
  IF existing<>p_doc_hash THEN RAISE EXCEPTION 'epoch % already has another payout',p_epoch USING ERRCODE='23505'; END IF;
  RETURN jsonb_build_object('status','exists');
 END IF;
 INSERT INTO public.lead_order_reward_epochs(epoch,mode,doc_hash,doc) VALUES(p_epoch,p_mode,p_doc_hash,p_doc);
 -- Shadow payouts are published, never recorded as paid, so shadow never
 -- shrinks what live would later owe.
 IF p_mode='live' THEN
  FOR p IN SELECT * FROM jsonb_array_elements(p_payments) LOOP
   INSERT INTO public.lead_order_reward_payments(epoch,order_id,miner_hotkey,paid_microusd)
   VALUES(p_epoch,(p->>'order_id')::UUID,p->>'miner_hotkey',(p->>'paid_microusd')::BIGINT);
   IF (SELECT sum(paid_microusd) FROM public.lead_order_reward_payments
       WHERE order_id=(p->>'order_id')::UUID AND miner_hotkey=p->>'miner_hotkey')
     >(SELECT owed_microusd FROM public.lead_order_rewards_owed
       WHERE order_id=(p->>'order_id')::UUID AND miner_hotkey=p->>'miner_hotkey') THEN
    RAISE EXCEPTION 'payout exceeds what is owed' USING ERRCODE='22023';
   END IF;
  END LOOP;
 END IF;
 RETURN jsonb_build_object('status','recorded');
END;
$payout703$;

CREATE OR REPLACE FUNCTION public.lead_order_reward_epoch_v1(p_epoch BIGINT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $epoch703$
 SELECT doc FROM public.lead_order_reward_epochs WHERE epoch=p_epoch;
$epoch703$;

-- Every lead_order function: owner, no PUBLIC execute, service-only execute.
DO $functions703$
DECLARE f REGPROCEDURE;
BEGIN
 FOR f IN SELECT p.oid::REGPROCEDURE FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
  WHERE n.nspname='public' AND p.proname LIKE 'lead\_order%\_v1' LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  IF f::TEXT NOT LIKE 'lead_order\_\_%' THEN
   EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f);
  END IF;
 END LOOP;
END;
$functions703$;

DO $restore703$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.lead_orders703_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.lead_orders703_acl)THEN
  RAISE EXCEPTION 'lead_orders703 schema ACL differs';END IF;
END;
$restore703$;
NOTIFY pgrst,'reload schema';
COMMIT;
