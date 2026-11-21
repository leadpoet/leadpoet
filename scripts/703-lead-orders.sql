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
  -- The miner's own spend cap for this key over any rolling 24 hours, USD 1 to
  -- 10,000. Enforced in lead_order_reserve_call_v1.
  daily_cap_microusd BIGINT NOT NULL CHECK(daily_cap_microusd BETWEEN 1000000 AND 10000000000),
  registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_key_revocations (
  key_id TEXT PRIMARY KEY REFERENCES public.lead_order_keys(key_id),
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 200),
  revoked_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- A miner's hotkey-signed acceptance of one version of the miner data terms.
-- The signature is over the request that accepted them (auth.signed_message:
-- METHOD, host, path, signed_at, body_sha256), so it can be re-verified later.
CREATE TABLE IF NOT EXISTS public.lead_order_miner_terms (
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  terms_version TEXT NOT NULL CHECK(terms_version ~ '^[A-Za-z0-9._-]{1,32}$'),
  -- sha256 of the exact terms text: an acceptance counts only for those words.
  terms_sha256 TEXT NOT NULL CHECK(terms_sha256 ~ '^[0-9a-f]{64}$'),
  signed_method TEXT NOT NULL CHECK(signed_method='POST'),
  signed_host TEXT NOT NULL CHECK(signed_host ~ '^[a-z0-9.-]{1,253}(:[0-9]{1,5})?$'),
  signed_path TEXT NOT NULL CHECK(signed_path='/lead-orders/miner-terms'),
  signed_at BIGINT NOT NULL CHECK(signed_at>0),
  body_sha256 TEXT NOT NULL CHECK(body_sha256 ~ '^[0-9a-f]{64}$'),
  signature TEXT NOT NULL CHECK(signature ~ '^(0x)?[0-9a-fA-F]{128}$'),
  accepted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(miner_hotkey,terms_sha256)
);

-- Suppressed people, as hashes only: sha256 of the lowercased email or of the
-- normalised LinkedIn URL. Raw values are never accepted or stored.
CREATE TABLE IF NOT EXISTS public.lead_order_suppressions (
  sequence BIGSERIAL NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK(kind IN ('email_sha256','linkedin_sha256')),
  hash TEXT NOT NULL CHECK(hash ~ '^[0-9a-f]{64}$'),
  added_at TIMESTAMPTZ NOT NULL,
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(kind,hash)
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
CREATE INDEX IF NOT EXISTS lead_order_calls_key_idx ON public.lead_order_calls(key_id,created_at);
CREATE INDEX IF NOT EXISTS lead_order_calls_miner_idx ON public.lead_order_calls(miner_hotkey,sequence);

CREATE TABLE IF NOT EXISTS public.lead_order_call_settlements (
  call_id UUID PRIMARY KEY REFERENCES public.lead_order_calls(call_id),
  status TEXT NOT NULL CHECK(status IN ('settled','uncertain')),
  cost_microusd BIGINT CHECK(cost_microusd IS NULL OR cost_microusd>=0),
  http_status INTEGER CHECK(http_status IS NULL OR http_status BETWEEN 100 AND 599),
  settled_epoch BIGINT NOT NULL CHECK(settled_epoch>=0),
  -- The provider's own id for the call (OpenRouter generation id, Deepline
  -- request id), kept for the miner's statement and reconcile only. Never
  -- published in a receipt.
  provider_request_id TEXT CHECK(provider_request_id IS NULL OR provider_request_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'),
  settled_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK((status='settled')=(cost_microusd IS NOT NULL))
);

-- A provider answer that arrived after its order finished (the call's own
-- row was already settled uncertain at finish). Never charged to the buyer;
-- a known cost is repaid to the miner through lead_order_late_rewards.
CREATE TABLE IF NOT EXISTS public.lead_order_late_settlements (
  call_id UUID PRIMARY KEY REFERENCES public.lead_order_calls(call_id),
  status TEXT NOT NULL CHECK(status IN ('settled','uncertain')),
  cost_microusd BIGINT CHECK(cost_microusd IS NULL OR cost_microusd>=0),
  http_status INTEGER CHECK(http_status IS NULL OR http_status BETWEEN 100 AND 599),
  provider_request_id TEXT CHECK(provider_request_id IS NULL OR provider_request_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'),
  received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK((status='settled')=(cost_microusd IS NOT NULL))
);

-- The exact cost of an uncertain OpenRouter call, read back later from the
-- provider's generation record on the same key. Never charged to the buyer.
CREATE TABLE IF NOT EXISTS public.lead_order_call_resolutions (
  call_id UUID PRIMARY KEY REFERENCES public.lead_order_calls(call_id),
  cost_microusd BIGINT NOT NULL CHECK(cost_microusd>=0),
  resolved_epoch BIGINT NOT NULL CHECK(resolved_epoch>=0),
  resolved_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- Every read-back attempt on an uncertain call, so a call that never answers
-- (a revoked or broken key, a generation that is gone) backs off and stops
-- instead of taking every pass.
CREATE TABLE IF NOT EXISTS public.lead_order_call_readbacks (
  call_id UUID NOT NULL REFERENCES public.lead_order_calls(call_id),
  attempt INTEGER NOT NULL CHECK(attempt BETWEEN 1 AND 100),
  outcome TEXT NOT NULL CHECK(outcome IN ('found','retry','terminal')),
  attempted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(call_id,attempt)
);

CREATE TABLE IF NOT EXISTS public.lead_order_rewards_owed (
  order_id UUID NOT NULL REFERENCES public.lead_order_finishes(order_id),
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  owed_microusd BIGINT NOT NULL CHECK(owed_microusd>0),
  PRIMARY KEY(order_id,miner_hotkey)
);

-- A miner's repayment for a call whose cost became known after its order's
-- receipt (a late answer or a resolution). One row per call, paid from the
-- lane like any reward, never charged to the buyer. owed is
-- min(cost, reserved) x (1 + premium), capped so the order's rewards stay
-- within charge x (1 + premium) / min_charge_ratio (the anti-farming bound);
-- 0 is recorded when nothing fits, so the shortfall stays visible.
CREATE TABLE IF NOT EXISTS public.lead_order_late_rewards (
  call_id UUID PRIMARY KEY REFERENCES public.lead_order_calls(call_id),
  sequence BIGSERIAL NOT NULL UNIQUE,
  order_id UUID NOT NULL REFERENCES public.lead_order_finishes(order_id),
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  cost_microusd BIGINT NOT NULL CHECK(cost_microusd>=0),
  owed_microusd BIGINT NOT NULL CHECK(owed_microusd>=0),
  settled_epoch BIGINT NOT NULL CHECK(settled_epoch>=0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS public.lead_order_receipts (
  order_id UUID PRIMARY KEY REFERENCES public.lead_order_finishes(order_id),
  receipt_hash TEXT NOT NULL UNIQUE CHECK(receipt_hash ~ '^sha256:[0-9a-f]{64}$'),
  receipt_doc JSONB NOT NULL,
  -- The exact call rows the Merkle root was built from, as published.
  calls_doc JSONB NOT NULL CHECK(jsonb_typeof(calls_doc)='array'),
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

CREATE TABLE IF NOT EXISTS public.lead_order_late_reward_payments (
  epoch BIGINT NOT NULL REFERENCES public.lead_order_reward_epochs(epoch),
  call_id UUID NOT NULL REFERENCES public.lead_order_late_rewards(call_id),
  paid_microusd BIGINT NOT NULL CHECK(paid_microusd>0),
  PRIMARY KEY(epoch,call_id)
);

-- Ownership, RLS, no table grants, append-only triggers (row and TRUNCATE).
DO $tables703$
DECLARE t TEXT;
BEGIN
 FOREACH t IN ARRAY ARRAY['lead_order_policies','lead_order_gamma_readings','lead_order_buyers',
   'lead_order_buyer_confirmations','lead_order_nonces','lead_orders','lead_order_claims',
   'lead_order_finishes','lead_order_keys','lead_order_key_revocations','lead_order_miner_terms',
   'lead_order_suppressions','lead_order_key_exhaustions',
   'lead_order_calls','lead_order_call_settlements','lead_order_late_settlements','lead_order_call_resolutions',
   'lead_order_call_readbacks',
   'lead_order_rewards_owed','lead_order_late_rewards','lead_order_receipts',
   'lead_order_reward_epochs','lead_order_reward_payments','lead_order_late_reward_payments'] LOOP
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
  RETURN jsonb_build_object('status','counter_backwards');
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
  -- A retry with the same id is the same order only if every field matches.
  IF EXISTS(SELECT 1 FROM public.lead_orders WHERE order_id=p_order_id AND hotkey=p_hotkey AND mode=p_mode
            AND leads_requested=p_leads AND request_doc=p_request_doc) THEN
   RETURN jsonb_build_object('status','exists');
  END IF;
  RETURN jsonb_build_object('status','conflict');
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
DECLARE o public.lead_orders; d TIMESTAMPTZ; attempt INTEGER;
BEGIN
 -- A concurrent claimer can commit a claim for the row this statement saw as
 -- free. The insert then skips it and the next queued order is tried.
 FOR attempt IN 1..20 LOOP
  o:=NULL;
  SELECT * INTO o FROM public.lead_orders q
   WHERE NOT EXISTS(SELECT 1 FROM public.lead_order_claims c WHERE c.order_id=q.order_id)
   ORDER BY q.sequence LIMIT 1 FOR UPDATE SKIP LOCKED;
  IF o.order_id IS NULL THEN RETURN NULL; END IF;
  d:=clock_timestamp()+make_interval(mins=>o.deadline_minutes);
  INSERT INTO public.lead_order_claims(order_id,token_hash,deadline) VALUES(o.order_id,p_token_hash,d)
  ON CONFLICT DO NOTHING;
  IF FOUND THEN
   RETURN jsonb_build_object('order_id',o.order_id,'hotkey',o.hotkey,'mode',o.mode,
    'leads_requested',o.leads_requested,'deadline',to_char(d AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'),
    'spend_ceiling_microusd',o.spend_ceiling_microusd,'request_doc',o.request_doc);
  END IF;
 END LOOP;
 RETURN NULL;
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
-- after the deadline, once the order is finished, and on a miner key whose
-- rolling 24-hour spend plus this reservation would pass the miner's own cap.
CREATE OR REPLACE FUNCTION public.lead_order_reserve_call_v1(
 p_call_id UUID,p_order_id UUID,p_provider TEXT,p_key_id TEXT,p_miner_hotkey TEXT,p_public_fingerprint TEXT,p_reserve BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reserve703$
DECLARE o public.lead_orders; c public.lead_order_claims; committed BIGINT; cap BIGINT; day_spend BIGINT;
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
 IF p_miner_hotkey IS NOT NULL THEN
  -- Match the full key binding and share revoke_key's row lock. Check
  -- revocation after acquiring it so a concurrent revoke cannot be missed.
  -- The lock also serialises the daily cap across orders.
  SELECT k.daily_cap_microusd INTO cap FROM public.lead_order_keys k
   WHERE k.key_id=p_key_id AND k.miner_hotkey=p_miner_hotkey
    AND k.provider=p_provider AND k.public_fingerprint=p_public_fingerprint FOR UPDATE;
  IF cap IS NULL OR EXISTS(SELECT 1 FROM public.lead_order_key_revocations WHERE key_id=p_key_id) THEN
   RETURN jsonb_build_object('status','key_unknown');
  END IF;
  SELECT coalesce(sum(CASE WHEN s.status='settled' THEN s.cost_microusd ELSE k.reserved_microusd END),0)::BIGINT
   INTO day_spend FROM public.lead_order_calls k LEFT JOIN public.lead_order_call_settlements s ON s.call_id=k.call_id
   WHERE k.key_id=p_key_id AND k.created_at>clock_timestamp()-interval '24 hours';
  IF day_spend+p_reserve>cap THEN RETURN jsonb_build_object('status','key_capped'); END IF;
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
 p_call_id UUID,p_status TEXT,p_cost BIGINT,p_http_status INTEGER,p_epoch BIGINT,p_request_id TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $settle703$
DECLARE v_order UUID;
BEGIN
 -- Share-lock the order so a settlement and the finish never interleave.
 SELECT o.order_id INTO v_order FROM public.lead_order_calls c JOIN public.lead_orders o ON o.order_id=c.order_id
  WHERE c.call_id=p_call_id FOR SHARE OF o;
 IF v_order IS NULL THEN RETURN jsonb_build_object('status','unknown'); END IF;
 IF EXISTS(SELECT 1 FROM public.lead_order_finishes WHERE order_id=v_order) THEN
  INSERT INTO public.lead_order_late_settlements(call_id,status,cost_microusd,http_status,provider_request_id)
  VALUES(p_call_id,p_status,CASE WHEN p_status='settled' THEN p_cost END,p_http_status,p_request_id)
  ON CONFLICT(call_id) DO NOTHING;
  RETURN jsonb_build_object('status','late');
 END IF;
 INSERT INTO public.lead_order_call_settlements(call_id,status,cost_microusd,http_status,settled_epoch,provider_request_id)
 VALUES(p_call_id,p_status,CASE WHEN p_status='settled' THEN p_cost END,p_http_status,p_epoch,p_request_id)
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
-- The runner's finish is refused while any call is open (calls_open); only
-- the overdue sweep (p_allow_open) closes open calls as uncertain.
CREATE OR REPLACE FUNCTION public.lead_order_finish_v1(
 p_order_id UUID,p_status TEXT,p_delivered INTEGER,p_delivery_url TEXT,p_failure_reason TEXT,p_epoch BIGINT,
 p_allow_open BOOLEAN)
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
  RETURN jsonb_build_object('status','delivered_out_of_range');
 END IF;
 -- A call open for less than the longest provider wait (240 s) plus a margin
 -- may still answer: the runner waits. An older one was orphaned (the broker
 -- stopped between reserve and settle) and settles uncertain below.
 IF NOT coalesce(p_allow_open,FALSE) AND EXISTS(SELECT 1 FROM public.lead_order_calls c WHERE c.order_id=p_order_id
   AND c.created_at>clock_timestamp()-interval '300 seconds'
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_call_settlements s WHERE s.call_id=c.call_id)) THEN
  RETURN jsonb_build_object('status','calls_open');
 END IF;
 -- A failed order delivers nothing and pays the minimum charge on spend so far.
 delivered:=CASE WHEN p_status='delivered' THEN p_delivered ELSE 0 END;
 -- Calls still open now (orphaned, or the overdue sweep) settle uncertain: never charged,
 -- and the rows are final before the receipt is built. The order row lock above
 -- also stops new reservations and waits out any settlement in progress.
 INSERT INTO public.lead_order_call_settlements(call_id,status,cost_microusd,http_status,settled_epoch)
 SELECT c.call_id,'uncertain',NULL,NULL,p_epoch FROM public.lead_order_calls c
 WHERE c.order_id=p_order_id AND NOT EXISTS(SELECT 1 FROM public.lead_order_call_settlements s WHERE s.call_id=c.call_id);
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
  'receipt',r.receipt_doc,'receipt_calls',r.calls_doc)
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

CREATE OR REPLACE FUNCTION public.lead_order_record_receipt_v1(p_order_id UUID,p_hash TEXT,p_doc JSONB,p_calls JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $receipt703$
DECLARE existing TEXT;
BEGIN
 SELECT receipt_hash INTO existing FROM public.lead_order_receipts WHERE order_id=p_order_id;
 IF existing IS NOT NULL THEN
  IF existing<>p_hash THEN RAISE EXCEPTION 'order already has another receipt' USING ERRCODE='23505'; END IF;
  RETURN jsonb_build_object('status','exists');
 END IF;
 INSERT INTO public.lead_order_receipts(order_id,receipt_hash,receipt_doc,calls_doc) VALUES(p_order_id,p_hash,p_doc,p_calls);
 RETURN jsonb_build_object('status','recorded');
END;
$receipt703$;

CREATE OR REPLACE FUNCTION public.lead_order_register_key_v1(
 p_key_id TEXT,p_provider TEXT,p_miner_hotkey TEXT,p_submission_id TEXT,p_ciphertext TEXT,p_internal_fp TEXT,p_public_fp TEXT,
 p_daily_cap BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $key703$
BEGIN
 -- One active key per miner and provider: a new key needs the old one revoked.
 IF EXISTS(SELECT 1 FROM public.lead_order_keys k WHERE k.miner_hotkey=p_miner_hotkey AND k.provider=p_provider
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_key_revocations r WHERE r.key_id=k.key_id)) THEN
  RETURN jsonb_build_object('status','active_key_exists');
 END IF;
 INSERT INTO public.lead_order_keys(key_id,provider,miner_hotkey,submission_id,ciphertext_b64,internal_fingerprint,
  public_fingerprint,daily_cap_microusd)
 VALUES(p_key_id,p_provider,p_miner_hotkey,p_submission_id,p_ciphertext,p_internal_fp,p_public_fp,p_daily_cap);
 RETURN jsonb_build_object('status','registered','key_id',p_key_id);
END;
$key703$;

CREATE OR REPLACE FUNCTION public.lead_order_revoke_key_v1(p_key_id TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $revoke703$
BEGIN
 -- Serialize with new reservations; already reserved calls can still settle.
 PERFORM 1 FROM public.lead_order_keys WHERE key_id=p_key_id FOR UPDATE;
 INSERT INTO public.lead_order_key_revocations(key_id,reason) VALUES(p_key_id,p_reason) ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status','revoked');
END;
$revoke703$;

-- Active keys. With a terms hash, only keys whose miner accepted exactly that
-- text (the broker's ranking); NULL lists every active key (revoke).
CREATE OR REPLACE FUNCTION public.lead_order_keys_v1(p_provider TEXT,p_terms_sha256 TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $keys703$
 SELECT coalesce(jsonb_agg(jsonb_build_object('key_id',k.key_id,'provider',k.provider,'miner_hotkey',k.miner_hotkey,
   'sequence',k.sequence,'ciphertext_b64',k.ciphertext_b64,'internal_fingerprint',k.internal_fingerprint) ORDER BY k.sequence),'[]'::JSONB)
 FROM public.lead_order_keys k
 WHERE k.provider=p_provider AND NOT EXISTS(SELECT 1 FROM public.lead_order_key_revocations r WHERE r.key_id=k.key_id)
  AND (p_terms_sha256 IS NULL OR EXISTS(SELECT 1 FROM public.lead_order_miner_terms t
       WHERE t.miner_hotkey=k.miner_hotkey AND t.terms_sha256=p_terms_sha256));
$keys703$;

CREATE OR REPLACE FUNCTION public.lead_order_accept_miner_terms_v1(
 p_hotkey TEXT,p_version TEXT,p_terms_sha256 TEXT,p_host TEXT,p_signed_at BIGINT,p_body_sha256 TEXT,p_signature TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $terms703$
BEGIN
 INSERT INTO public.lead_order_miner_terms(miner_hotkey,terms_version,terms_sha256,signed_method,signed_host,signed_path,
  signed_at,body_sha256,signature)
 VALUES(p_hotkey,p_version,p_terms_sha256,'POST',p_host,'/lead-orders/miner-terms',p_signed_at,p_body_sha256,p_signature)
 ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status',CASE WHEN FOUND THEN 'accepted' ELSE 'exists' END,
  'accepted_at',(SELECT accepted_at FROM public.lead_order_miner_terms WHERE miner_hotkey=p_hotkey AND terms_sha256=p_terms_sha256));
END;
$terms703$;

CREATE OR REPLACE FUNCTION public.lead_order_miner_terms_v1(p_hotkey TEXT,p_terms_sha256 TEXT)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $hasterms703$
 SELECT EXISTS(SELECT 1 FROM public.lead_order_miner_terms WHERE miner_hotkey=p_hotkey AND terms_sha256=p_terms_sha256);
$hasterms703$;

-- Append hashes; a hash already on the list keeps its first row.
CREATE OR REPLACE FUNCTION public.lead_order_add_suppressions_v1(p_rows JSONB)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $supadd703$
DECLARE v_added INTEGER;
BEGIN
 IF jsonb_typeof(p_rows)<>'array' OR jsonb_array_length(p_rows) NOT BETWEEN 1 AND 1000 THEN
  RAISE EXCEPTION 'suppressions must be 1 to 1000 rows';
 END IF;
 INSERT INTO public.lead_order_suppressions(kind,hash,added_at)
 SELECT r->>'kind',r->>'hash',(r->>'added_at')::TIMESTAMPTZ FROM jsonb_array_elements(p_rows) r
 ON CONFLICT DO NOTHING;
 GET DIAGNOSTICS v_added=ROW_COUNT;
 RETURN jsonb_build_object('received',jsonb_array_length(p_rows),'added',v_added);
END;
$supadd703$;

CREATE OR REPLACE FUNCTION public.lead_order_suppressions_v1(p_since BIGINT,p_limit INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $suplist703$
 SELECT coalesce(jsonb_agg(jsonb_build_object('sequence',s.sequence,'kind',s.kind,'hash',s.hash,
   'added_at',to_char(s.added_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"')) ORDER BY s.sequence),'[]'::JSONB)
 FROM (SELECT * FROM public.lead_order_suppressions WHERE sequence>greatest(p_since,0)
       ORDER BY sequence LIMIT least(greatest(p_limit,1),1000)) s;
$suplist703$;

-- Settled spend per key over the ranking window, from live orders only:
-- staff and shadow spend never moves the live ranking.
CREATE OR REPLACE FUNCTION public.lead_order_key_spend_v1(p_since_epoch BIGINT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $spend703$
 SELECT coalesce(jsonb_object_agg(key_id,total),'{}'::JSONB) FROM (
  SELECT c.key_id,sum(s.cost_microusd)::BIGINT total FROM public.lead_order_calls c
  JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id
  JOIN public.lead_orders o ON o.order_id=c.order_id
  WHERE s.status='settled' AND s.settled_epoch>=p_since_epoch AND o.mode='live' GROUP BY c.key_id) t;
$spend703$;

CREATE OR REPLACE FUNCTION public.lead_order_owed_v1(p_through_epoch BIGINT,p_mode TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $owed703$
 SELECT coalesce(jsonb_agg(e ORDER BY (e->>'settled_epoch')::BIGINT,(e->>'settled_sequence')::BIGINT,
   e->>'miner_hotkey',coalesce(e->>'call_id','')),'[]'::JSONB) FROM (
  SELECT jsonb_build_object('order_id',w.order_id,'miner_hotkey',w.miner_hotkey,'call_id',NULL,
   'settled_epoch',f.settled_epoch,'settled_sequence',f.sequence,'owed_microusd',w.owed_microusd,
   'paid_microusd',coalesce((SELECT sum(p.paid_microusd) FROM public.lead_order_reward_payments p
     WHERE p.order_id=w.order_id AND p.miner_hotkey=w.miner_hotkey),0)) e
  FROM public.lead_order_rewards_owed w JOIN public.lead_order_finishes f ON f.order_id=w.order_id
  JOIN public.lead_orders o ON o.order_id=w.order_id
  -- Per mode: a live payout reads live orders only, so live never pays a
  -- shadow backlog; a shadow payout publishes shadow orders' rewards only.
  WHERE f.settled_epoch<=p_through_epoch AND o.mode=p_mode AND p_mode IN ('shadow','live')
  UNION ALL
  -- Late repayments, one entry per call, payable from the epoch they were recorded.
  SELECT jsonb_build_object('order_id',l.order_id,'miner_hotkey',l.miner_hotkey,'call_id',l.call_id,
   'settled_epoch',l.settled_epoch,'settled_sequence',l.sequence,'owed_microusd',l.owed_microusd,
   'paid_microusd',coalesce((SELECT sum(p.paid_microusd) FROM public.lead_order_late_reward_payments p
     WHERE p.call_id=l.call_id),0))
  FROM public.lead_order_late_rewards l JOIN public.lead_orders o ON o.order_id=l.order_id
  WHERE l.settled_epoch<=p_through_epoch AND l.owed_microusd>0 AND o.mode=p_mode AND p_mode IN ('shadow','live')) t;
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
   IF NOT EXISTS(SELECT 1 FROM public.lead_orders WHERE order_id=(p->>'order_id')::UUID AND mode='live') THEN
    RAISE EXCEPTION 'only live orders are paid' USING ERRCODE='22023';
   END IF;
   IF p->>'call_id' IS NOT NULL THEN
    IF NOT EXISTS(SELECT 1 FROM public.lead_order_late_rewards WHERE call_id=(p->>'call_id')::UUID
      AND order_id=(p->>'order_id')::UUID AND miner_hotkey=p->>'miner_hotkey') THEN
     RAISE EXCEPTION 'late payment names another order or miner' USING ERRCODE='22023';
    END IF;
    INSERT INTO public.lead_order_late_reward_payments(epoch,call_id,paid_microusd)
    VALUES(p_epoch,(p->>'call_id')::UUID,(p->>'paid_microusd')::BIGINT);
    IF (SELECT sum(paid_microusd) FROM public.lead_order_late_reward_payments WHERE call_id=(p->>'call_id')::UUID)
      >(SELECT owed_microusd FROM public.lead_order_late_rewards WHERE call_id=(p->>'call_id')::UUID) THEN
     RAISE EXCEPTION 'payout exceeds what is owed' USING ERRCODE='22023';
    END IF;
    CONTINUE;
   END IF;
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

CREATE OR REPLACE FUNCTION public.lead_order_policy_v1(p_version INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $policyget703$
 SELECT policy_doc FROM public.lead_order_policies WHERE version=p_version;
$policyget703$;

-- Finished orders still missing their receipt, oldest first.
CREATE OR REPLACE FUNCTION public.lead_order_unsettled_v1(p_limit INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $unsettled703$
 SELECT coalesce(jsonb_agg(order_id ORDER BY sequence),'[]'::JSONB) FROM (
  SELECT f.order_id,f.sequence FROM public.lead_order_finishes f
  WHERE NOT EXISTS(SELECT 1 FROM public.lead_order_receipts r WHERE r.order_id=f.order_id)
  ORDER BY f.sequence LIMIT greatest(1,least(p_limit,500))) t;
$unsettled703$;

CREATE OR REPLACE FUNCTION public.lead_order_reward_epoch_v1(p_epoch BIGINT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $epoch703$
 SELECT doc FROM public.lead_order_reward_epochs WHERE epoch=p_epoch;
$epoch703$;

-- Uncertain OpenRouter calls on finished live orders that have a provider
-- generation id and an active miner key: the reconcile pass reads their exact
-- cost back. Calls older than 7 days stay uncertain (bounded, fail closed).
CREATE OR REPLACE FUNCTION public.lead_order_uncertain_calls_v1(p_limit INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $uncertain703$
 SELECT coalesce(jsonb_agg(jsonb_build_object('call_id',t.call_id,'order_id',t.order_id,'provider',t.provider,
   'key_id',t.key_id,'miner_hotkey',t.miner_hotkey,'sequence',t.key_sequence,'ciphertext_b64',t.ciphertext_b64,
   'internal_fingerprint',t.internal_fingerprint,'provider_request_id',t.request_id,'attempts',t.attempts)
   ORDER BY t.last_attempt NULLS FIRST,t.sequence),'[]'::JSONB)
 FROM (SELECT c.call_id,c.order_id,c.provider,c.key_id,c.miner_hotkey,c.sequence,k.sequence key_sequence,
        k.ciphertext_b64,k.internal_fingerprint,coalesce(l.provider_request_id,s.provider_request_id) request_id,
        coalesce(a.attempts,0) attempts,a.last_attempt
  FROM public.lead_order_calls c
  LEFT JOIN LATERAL (SELECT count(*)::INTEGER attempts,max(attempted_at) last_attempt,bool_or(outcome='terminal') terminal
    FROM public.lead_order_call_readbacks b WHERE b.call_id=c.call_id) a ON TRUE
  JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id AND s.status='uncertain'
  JOIN public.lead_order_finishes f ON f.order_id=c.order_id
  JOIN public.lead_orders o ON o.order_id=c.order_id AND o.mode='live'
  JOIN public.lead_order_keys k ON k.key_id=c.key_id
  LEFT JOIN public.lead_order_late_settlements l ON l.call_id=c.call_id
  WHERE c.provider='openrouter' AND c.created_at>clock_timestamp()-interval '7 days'
   AND coalesce(l.status,'uncertain')='uncertain'
   AND coalesce(l.provider_request_id,s.provider_request_id) IS NOT NULL
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_key_revocations r WHERE r.key_id=c.key_id)
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_call_resolutions x WHERE x.call_id=c.call_id)
   -- At most 6 tries, never after a terminal answer, waiting 10 min x 2^(tries-1)
   -- after each retry (10, 20, 40, 80, 160 min).
   AND NOT coalesce(a.terminal,FALSE) AND coalesce(a.attempts,0)<6
   AND (a.last_attempt IS NULL
     OR a.last_attempt<=clock_timestamp()-interval '10 minutes'*power(2,a.attempts-1))
  ORDER BY a.last_attempt NULLS FIRST,c.sequence LIMIT greatest(1,least(p_limit,100))) t;
$uncertain703$;

-- Record one read-back attempt and its outcome (append-only).
CREATE OR REPLACE FUNCTION public.lead_order_record_readback_v1(p_call_id UUID,p_outcome TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $readback703$
DECLARE n INTEGER;
BEGIN
 IF p_outcome IS NULL OR p_outcome NOT IN ('found','retry','terminal') OR NOT EXISTS(
   SELECT 1 FROM public.lead_order_call_settlements s WHERE s.call_id=p_call_id AND s.status='uncertain') THEN
  RETURN jsonb_build_object('status','not_recordable');
 END IF;
 PERFORM 1 FROM public.lead_order_calls WHERE call_id=p_call_id FOR UPDATE;
 SELECT coalesce(max(attempt),0)+1 INTO n FROM public.lead_order_call_readbacks WHERE call_id=p_call_id;
 IF n>100 THEN
  RETURN jsonb_build_object('status','not_recordable');
 END IF;
 INSERT INTO public.lead_order_call_readbacks(call_id,attempt,outcome) VALUES(p_call_id,n,p_outcome);
 RETURN jsonb_build_object('status','recorded','attempt',n);
END;
$readback703$;

-- Record the read-back cost of one uncertain call. Never charged to the buyer.
CREATE OR REPLACE FUNCTION public.lead_order_resolve_call_v1(p_call_id UUID,p_cost BIGINT,p_epoch BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $resolve703$
BEGIN
 IF p_cost IS NULL OR p_cost<0 OR NOT EXISTS(
   SELECT 1 FROM public.lead_order_calls c
   JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id AND s.status='uncertain'
   JOIN public.lead_order_finishes f ON f.order_id=c.order_id
   WHERE c.call_id=p_call_id AND NOT EXISTS(SELECT 1 FROM public.lead_order_late_settlements l
     WHERE l.call_id=c.call_id AND l.status='settled')) THEN
  RETURN jsonb_build_object('status','not_resolvable');
 END IF;
 INSERT INTO public.lead_order_call_resolutions(call_id,cost_microusd,resolved_epoch) VALUES(p_call_id,p_cost,p_epoch)
 ON CONFLICT(call_id) DO NOTHING;
 RETURN jsonb_build_object('status',CASE WHEN FOUND THEN 'resolved' ELSE 'exists' END);
END;
$resolve703$;

-- Turn late known costs into late rewards, once per call, after the order's
-- receipt (so its on-time rewards are recorded). owed = min(cost, reserved) x
-- (1 + premium), capped so all rewards on the order stay within
-- charge x (1 + premium) / min_charge_ratio. Live orders and miner keys only.
CREATE OR REPLACE FUNCTION public.lead_order_reward_late_v1(p_epoch BIGINT,p_limit INTEGER)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $late703$
DECLARE r RECORD; prem BIGINT; reward BIGINT; room BIGINT; n INTEGER:=0;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('lead_order_late_rewards',0));
 FOR r IN SELECT c.call_id,c.order_id,c.miner_hotkey,c.reserved_microusd,coalesce(l.cost_microusd,x.cost_microusd) cost,
   o.policy_version,o.min_charge_ratio_ppm,f.charge_microusd
  FROM public.lead_order_calls c
  JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id AND s.status='uncertain'
  JOIN public.lead_orders o ON o.order_id=c.order_id AND o.mode='live'
  JOIN public.lead_order_finishes f ON f.order_id=c.order_id
  JOIN public.lead_order_receipts rc ON rc.order_id=c.order_id
  LEFT JOIN public.lead_order_late_settlements l ON l.call_id=c.call_id AND l.status='settled'
  LEFT JOIN public.lead_order_call_resolutions x ON x.call_id=c.call_id
  WHERE c.miner_hotkey IS NOT NULL AND coalesce(l.cost_microusd,x.cost_microusd) IS NOT NULL
   AND NOT EXISTS(SELECT 1 FROM public.lead_order_late_rewards w WHERE w.call_id=c.call_id)
  ORDER BY c.sequence LIMIT greatest(1,least(p_limit,500))
 LOOP
  SELECT trunc((policy_doc->>'miner_premium')::NUMERIC*1000000)::BIGINT INTO prem
   FROM public.lead_order_policies WHERE version=r.policy_version;
  reward:=div(LEAST(r.cost,r.reserved_microusd)::NUMERIC*(1000000+prem),1000000)::BIGINT;
  room:=div(r.charge_microusd::NUMERIC*(1000000+prem),r.min_charge_ratio_ppm)::BIGINT
   -coalesce((SELECT sum(owed_microusd) FROM public.lead_order_rewards_owed WHERE order_id=r.order_id),0)
   -coalesce((SELECT sum(owed_microusd) FROM public.lead_order_late_rewards WHERE order_id=r.order_id),0);
  INSERT INTO public.lead_order_late_rewards(call_id,order_id,miner_hotkey,cost_microusd,owed_microusd,settled_epoch)
  VALUES(r.call_id,r.order_id,r.miner_hotkey,r.cost,GREATEST(0,LEAST(reward,room)),p_epoch);
  n:=n+1;
 END LOOP;
 RETURN jsonb_build_object('recorded',n);
END;
$late703$;

-- A miner's own statement: every call on the miner's keys after p_since (by
-- call sequence, at most 500), plus all-time totals per key. The provider
-- request id is shown here only, never in a public receipt.
CREATE OR REPLACE FUNCTION public.lead_order_statement_v1(p_hotkey TEXT,p_since BIGINT,p_limit INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $statement703$
 WITH mine AS (
  SELECT c.*,s.status settle_status,s.cost_microusd settled_cost,s.settled_at,
   coalesce(l.provider_request_id,s.provider_request_id) request_id,
   coalesce(CASE WHEN l.status='settled' THEN l.cost_microusd END,x.cost_microusd) late_cost,
   coalesce(CASE WHEN l.status='settled' THEN l.received_at END,x.resolved_at) late_at,
   rc.receipt_hash,w.owed_microusd order_reward,lr.owed_microusd late_reward
  FROM public.lead_order_calls c
  LEFT JOIN public.lead_order_call_settlements s ON s.call_id=c.call_id
  LEFT JOIN public.lead_order_late_settlements l ON l.call_id=c.call_id
  LEFT JOIN public.lead_order_call_resolutions x ON x.call_id=c.call_id
  LEFT JOIN public.lead_order_receipts rc ON rc.order_id=c.order_id
  LEFT JOIN public.lead_order_rewards_owed w ON w.order_id=c.order_id AND w.miner_hotkey=c.miner_hotkey
  LEFT JOIN public.lead_order_late_rewards lr ON lr.call_id=c.call_id
  WHERE c.miner_hotkey=p_hotkey),
 page AS (SELECT * FROM mine WHERE sequence>greatest(p_since,0) ORDER BY sequence LIMIT least(greatest(p_limit,1),500))
 SELECT jsonb_build_object(
  'calls',(SELECT coalesce(jsonb_agg(jsonb_build_object('sequence',sequence,'call_id',call_id,'order_id',order_id,
    'key_id',key_id,'provider',provider,'reserved_microusd',reserved_microusd,'status',coalesce(settle_status,'open'),
    'cost_microusd',settled_cost,
    'excess_microusd',CASE WHEN settled_cost>reserved_microusd THEN settled_cost-reserved_microusd ELSE 0 END,
    'settled_at',to_char(settled_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'),
    'provider_request_id',request_id,'late_cost_microusd',late_cost,
    'late_at',to_char(late_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS"Z"'),'late_reward_microusd',late_reward,
    'receipt_hash',receipt_hash,'order_reward_microusd',order_reward) ORDER BY sequence),'[]'::JSONB) FROM page),
  'keys',(SELECT coalesce(jsonb_agg(jsonb_build_object('key_id',key_id,'provider',provider,'calls',calls,
    'reserved_microusd',reserved,'settled_microusd',settled,'excess_microusd',excess,'open_calls',open_calls,
    'uncertain_calls',uncertain_calls,'late_cost_microusd',late_cost,'late_reward_microusd',late_reward) ORDER BY key_id),'[]'::JSONB)
   FROM (SELECT key_id,provider,count(*) calls,sum(reserved_microusd) reserved,
     coalesce(sum(settled_cost),0) settled,
     coalesce(sum(CASE WHEN settled_cost>reserved_microusd THEN settled_cost-reserved_microusd ELSE 0 END),0) excess,
     count(*) FILTER (WHERE settle_status IS NULL) open_calls,count(*) FILTER (WHERE settle_status='uncertain') uncertain_calls,
     coalesce(sum(late_cost),0) late_cost,coalesce(sum(late_reward),0) late_reward
    FROM mine GROUP BY key_id,provider) k),
  'rewards_owed_microusd',coalesce((SELECT sum(owed_microusd) FROM public.lead_order_rewards_owed WHERE miner_hotkey=p_hotkey),0)
    +coalesce((SELECT sum(owed_microusd) FROM public.lead_order_late_rewards WHERE miner_hotkey=p_hotkey),0),
  'rewards_paid_microusd',coalesce((SELECT sum(paid_microusd) FROM public.lead_order_reward_payments WHERE miner_hotkey=p_hotkey),0)
    +coalesce((SELECT sum(p.paid_microusd) FROM public.lead_order_late_reward_payments p
      JOIN public.lead_order_late_rewards l ON l.call_id=p.call_id WHERE l.miner_hotkey=p_hotkey),0),
  'next_since',coalesce((SELECT max(sequence) FROM page),greatest(p_since,0)),
  'more',(SELECT count(*) FROM page)=least(greatest(p_limit,1),500));
$statement703$;

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
