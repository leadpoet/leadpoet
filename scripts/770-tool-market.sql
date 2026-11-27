-- Gamma tool market (docs/gamma-tool-market.md). Off by default: nothing
-- calls these functions unless LEADPOET_TOOL_MARKET_MODE is live. Miners
-- list vendor keys sealed to the key vault with a price per unit; buyers open
-- a tab against their Gamma balance and each call goes to the cheapest
-- eligible key. Every table is append-only (no UPDATE, DELETE or TRUNCATE):
-- state moves forward by inserting asks, events, closures and settlements.
-- RLS is on with no policy and no table grant; the Arena service role reaches
-- rows only through the SECURITY DEFINER functions below, as 707 does.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
CREATE TEMP TABLE tool_market770_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $prereq770$
BEGIN
 IF to_regprocedure('public.lab_arena_append_only_v1()') IS NULL
  OR to_regrole('lab_arena_owner') IS NULL OR to_regrole('lab_arena_service') IS NULL THEN
  RAISE EXCEPTION 'the tool market needs the Lab Arena roles and append-only trigger (179)' USING ERRCODE='55000';
 END IF;
END;
$prereq770$;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE public.tool_market_gamma_readings (
  reading_id BIGSERIAL PRIMARY KEY,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  block_number BIGINT NOT NULL CHECK(block_number>=0),
  block_hash TEXT NOT NULL CHECK(block_hash ~ '^0x[0-9a-f]{64}$'),
  spent_units NUMERIC(39,0) NOT NULL CHECK(spent_units>=0),
  available BOOLEAN NOT NULL,
  read_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_gamma_readings_hotkey ON public.tool_market_gamma_readings(hotkey,block_number DESC,reading_id DESC);

CREATE TABLE public.tool_market_nonces (
  hotkey TEXT NOT NULL,
  signature_hash TEXT NOT NULL CHECK(signature_hash ~ '^sha256:[0-9a-f]{64}$'),
  signed_at BIGINT NOT NULL,
  used_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(hotkey,signature_hash)
);

-- A signed acceptance of the terms text whose sha256 the gateway is
-- configured with. Kept whole so it can be verified again later.
CREATE TABLE public.tool_market_terms (
  role TEXT NOT NULL CHECK(role IN ('buyer','miner')),
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  terms_sha256 TEXT NOT NULL CHECK(terms_sha256 ~ '^[0-9a-f]{64}$'),
  terms_version TEXT NOT NULL CHECK(terms_version ~ '^[A-Za-z0-9._-]{1,32}$'),
  host TEXT NOT NULL CHECK(length(host) BETWEEN 1 AND 253),
  signed_at BIGINT NOT NULL,
  body_sha256 TEXT NOT NULL CHECK(body_sha256 ~ '^[0-9a-f]{64}$'),
  signature TEXT NOT NULL CHECK(length(signature) BETWEEN 128 AND 130),
  accepted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  PRIMARY KEY(role,hotkey,terms_sha256)
);

-- ``envelope`` is the miner's key sealed to the vault; only the vault can
-- open it. ``key_fingerprint`` is the vault's public name for the key.
-- ``signed_body`` is the register request as the miner's hotkey signed it,
-- with their attestation that the key is theirs to offer; ``host``,
-- ``signed_at`` and ``signature`` let anyone verify it again.
CREATE TABLE public.tool_market_keys (
  key_id UUID PRIMARY KEY,
  sequence BIGSERIAL NOT NULL UNIQUE,
  tool TEXT NOT NULL CHECK(tool ~ '^[a-z_]{1,40}$'),
  miner_hotkey TEXT NOT NULL CHECK(miner_hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  envelope TEXT NOT NULL CHECK(envelope LIKE 'lov1.%' AND length(envelope)<=8192),
  key_fingerprint TEXT NOT NULL CHECK(key_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
  daily_cap_microusd BIGINT NOT NULL CHECK(daily_cap_microusd>0),
  host TEXT NOT NULL CHECK(length(host) BETWEEN 1 AND 253),
  signed_at BIGINT NOT NULL,
  signed_body TEXT NOT NULL CHECK(length(signed_body) BETWEEN 2 AND 65536),
  signature TEXT NOT NULL CHECK(length(signature) BETWEEN 128 AND 130),
  registered_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_keys_tool ON public.tool_market_keys(tool,sequence);
CREATE INDEX tool_market_keys_fingerprint ON public.tool_market_keys(key_fingerprint);
CREATE INDEX tool_market_keys_miner ON public.tool_market_keys(miner_hotkey,sequence);

CREATE TABLE public.tool_market_key_revocations (
  key_id UUID PRIMARY KEY REFERENCES public.tool_market_keys(key_id),
  reason TEXT NOT NULL CHECK(reason ~ '^[a-z_]{1,64}$'),
  revoked_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- A key's health: a vault check (valid, unchecked when the vendor has no free
-- check, or invalid) or a call that showed the key unusable (failed). The
-- latest event decides whether the key trades.
CREATE TABLE public.tool_market_key_events (
  event_id BIGSERIAL PRIMARY KEY,
  key_id UUID NOT NULL REFERENCES public.tool_market_keys(key_id),
  status TEXT NOT NULL CHECK(status IN ('valid','unchecked','invalid','failed')),
  reason TEXT CHECK(reason ~ '^[a-z_]{1,64}$'),
  balance NUMERIC CHECK(balance IS NULL OR balance>0),
  recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_key_events_key ON public.tool_market_key_events(key_id,event_id DESC);

-- The latest ask is the key's price; a NULL ask withdraws it.
CREATE TABLE public.tool_market_asks (
  ask_sequence BIGSERIAL PRIMARY KEY,
  key_id UUID NOT NULL REFERENCES public.tool_market_keys(key_id),
  ask_microusd BIGINT CHECK(ask_microusd IS NULL OR ask_microusd BETWEEN 1 AND 10000000),
  posted_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_asks_key ON public.tool_market_asks(key_id,ask_sequence DESC);

-- A buyer's budget, held against their Gamma balance while open. A tab is
-- open until it is closed or expires; then only what its calls committed
-- stays held.
CREATE TABLE public.tool_market_tabs (
  tab_id UUID PRIMARY KEY,
  hotkey TEXT NOT NULL CHECK(hotkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'),
  token_hash TEXT NOT NULL UNIQUE CHECK(token_hash ~ '^sha256:[0-9a-f]{64}$'),
  budget_microusd BIGINT NOT NULL CHECK(budget_microusd>0),
  expires_at TIMESTAMPTZ NOT NULL,
  opened_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_tabs_hotkey ON public.tool_market_tabs(hotkey,opened_at);

CREATE TABLE public.tool_market_tab_closures (
  tab_id UUID PRIMARY KEY REFERENCES public.tool_market_tabs(tab_id),
  closed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE public.tool_market_calls (
  call_id UUID PRIMARY KEY,
  sequence BIGSERIAL NOT NULL UNIQUE,
  tab_id UUID NOT NULL REFERENCES public.tool_market_tabs(tab_id),
  tool TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation ~ '^[a-z_]{1,40}$'),
  key_id UUID NOT NULL REFERENCES public.tool_market_keys(key_id),
  miner_hotkey TEXT NOT NULL,
  ask_microusd BIGINT NOT NULL CHECK(ask_microusd>0),
  reserve_microusd BIGINT NOT NULL CHECK(reserve_microusd>0),
  -- The most the miner can be owed for this call; the daily cap counts it.
  owed_cap_microusd BIGINT NOT NULL CHECK(owed_cap_microusd>=0),
  opened_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX tool_market_calls_tab ON public.tool_market_calls(tab_id);
CREATE INDEX tool_market_calls_key ON public.tool_market_calls(key_id,opened_at);

-- settled: the vendor answered and its usage is known. uncertain: it may
-- have charged the key but the usage is unknown (5xx, timeout, no cost in
-- the answer); the buyer pays nothing and the miner bears it. skipped: the
-- request was not sent, or the key failed and the next one was tried.
CREATE TABLE public.tool_market_call_settlements (
  call_id UUID PRIMARY KEY REFERENCES public.tool_market_calls(call_id),
  status TEXT NOT NULL CHECK(status IN ('settled','uncertain','skipped')),
  http_status INTEGER CHECK(http_status IS NULL OR http_status BETWEEN 100 AND 599),
  units BIGINT NOT NULL CHECK(units>=0),
  owed_microusd BIGINT NOT NULL CHECK(owed_microusd>=0),
  charge_microusd BIGINT NOT NULL CHECK(charge_microusd>=owed_microusd),
  epoch BIGINT NOT NULL CHECK(epoch>=0),
  settled_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  CHECK(status='settled' OR (units=0 AND owed_microusd=0 AND charge_microusd=0))
);
CREATE INDEX tool_market_call_settlements_epoch ON public.tool_market_call_settlements(epoch);

DO $tables770$
DECLARE t TEXT;
BEGIN
 FOREACH t IN ARRAY ARRAY['tool_market_gamma_readings','tool_market_nonces','tool_market_terms','tool_market_keys',
   'tool_market_key_revocations','tool_market_key_events','tool_market_asks','tool_market_tabs',
   'tool_market_tab_closures','tool_market_calls','tool_market_call_settlements'] LOOP
  EXECUTE format('ALTER TABLE public.%I OWNER TO lab_arena_owner',t);
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',t);
  EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1()',t||'_append_only',t);
  EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON public.%I FOR EACH STATEMENT EXECUTE FUNCTION public.lab_arena_append_only_v1()',t||'_no_truncate',t);
 END LOOP;
END;
$tables770$;
ALTER SEQUENCE public.tool_market_gamma_readings_reading_id_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_keys_sequence_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_key_events_event_id_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_asks_ask_sequence_seq OWNER TO lab_arena_owner;
ALTER SEQUENCE public.tool_market_calls_sequence_seq OWNER TO lab_arena_owner;

-- ---------------------------------------------------------------------------
-- Internal helpers (double underscore: never granted)
-- ---------------------------------------------------------------------------

-- A call left open this long is counted as uncertain: it holds nothing and
-- can no longer settle.
CREATE FUNCTION public.tool_market__open_seconds_v1() RETURNS INTEGER LANGUAGE sql IMMUTABLE AS $$ SELECT 600 $$;

-- What a call holds: its charge once settled, its reserve while young and
-- open, nothing after that.
CREATE FUNCTION public.tool_market__held_v1(p_call public.tool_market_calls)
RETURNS BIGINT LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $held770$
 SELECT coalesce((SELECT s.charge_microusd FROM public.tool_market_call_settlements s WHERE s.call_id=p_call.call_id),
  CASE WHEN p_call.opened_at>clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1())
   THEN p_call.reserve_microusd ELSE 0 END);
$held770$;

CREATE FUNCTION public.tool_market__committed_v1(p_tab_id UUID)
RETURNS BIGINT LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $committed770$
 SELECT coalesce(sum(public.tool_market__held_v1(c)),0)::BIGINT FROM public.tool_market_calls c WHERE c.tab_id=p_tab_id;
$committed770$;

CREATE FUNCTION public.tool_market__tab_open_v1(p_tab public.tool_market_tabs)
RETURNS BOOLEAN LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $open770$
 SELECT p_tab.expires_at>clock_timestamp()
  AND NOT EXISTS(SELECT 1 FROM public.tool_market_tab_closures x WHERE x.tab_id=p_tab.tab_id);
$open770$;

-- What a key has earned or holds over the last 24 h, in owed (miner) terms:
-- settled calls count what they owe, young open calls the most they can owe.
CREATE FUNCTION public.tool_market__key_day_v1(p_key_id UUID)
RETURNS BIGINT LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $day770$
 SELECT coalesce(sum(coalesce(s.owed_microusd,
   CASE WHEN c.opened_at>clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1())
    THEN c.owed_cap_microusd ELSE 0 END)),0)::BIGINT
 FROM public.tool_market_calls c LEFT JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
 WHERE c.key_id=p_key_id AND c.opened_at>clock_timestamp()-interval '24 hours';
$day770$;

CREATE FUNCTION public.tool_market__key_active_v1(p_key_id UUID)
RETURNS BOOLEAN LANGUAGE sql STABLE SET search_path=pg_catalog,public AS $active770$
 SELECT EXISTS(SELECT 1 FROM public.tool_market_keys k WHERE k.key_id=p_key_id)
  AND NOT EXISTS(SELECT 1 FROM public.tool_market_key_revocations r WHERE r.key_id=p_key_id);
$active770$;

CREATE FUNCTION public.tool_market__lock_v1(p_kind TEXT,p_value TEXT)
RETURNS VOID LANGUAGE sql SET search_path=pg_catalog,public AS $lock770$
 SELECT pg_advisory_xact_lock(hashtextextended('tool_market:'||p_kind||':'||p_value,0));
$lock770$;

-- ---------------------------------------------------------------------------
-- Buyers: balance, terms, tabs
-- ---------------------------------------------------------------------------

CREATE FUNCTION public.tool_market_balance_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $balance770$
DECLARE r public.tool_market_gamma_readings; held BIGINT; credit BIGINT;
BEGIN
 SELECT * INTO r FROM public.tool_market_gamma_readings WHERE hotkey=p_hotkey
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 SELECT coalesce(sum(CASE WHEN public.tool_market__tab_open_v1(t) THEN t.budget_microusd
   ELSE public.tool_market__committed_v1(t.tab_id) END),0)::BIGINT INTO held
  FROM public.tool_market_tabs t WHERE t.hotkey=p_hotkey;
 -- 10^9 Gamma units per USD: 1000 units per micro-USD, rounded down.
 credit:=CASE WHEN r.available THEN floor(r.spent_units/1000)::BIGINT ELSE 0 END;
 RETURN jsonb_build_object('available',coalesce(r.available,FALSE),'block_number',r.block_number,
  'credit_microusd',coalesce(credit,0),'held_microusd',held,'balance_microusd',coalesce(credit,0)-held);
END;
$balance770$;

CREATE FUNCTION public.tool_market_record_gamma_reading_v1(
 p_hotkey TEXT,p_block_number BIGINT,p_block_hash TEXT,p_units NUMERIC,p_available BOOLEAN)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reading770$
DECLARE prior public.tool_market_gamma_readings;
BEGIN
 SELECT * INTO prior FROM public.tool_market_gamma_readings WHERE hotkey=p_hotkey AND available
  ORDER BY block_number DESC,reading_id DESC LIMIT 1;
 -- The spent counter only grows; a smaller value is a broken read.
 IF p_available AND prior.reading_id IS NOT NULL AND p_block_number>=prior.block_number
  AND p_units<prior.spent_units THEN
  RETURN jsonb_build_object('status','counter_backwards');
 END IF;
 INSERT INTO public.tool_market_gamma_readings(hotkey,block_number,block_hash,spent_units,available)
 VALUES(p_hotkey,p_block_number,lower(p_block_hash),CASE WHEN p_available THEN p_units ELSE 0 END,p_available);
 RETURN public.tool_market_balance_v1(p_hotkey);
END;
$reading770$;

CREATE FUNCTION public.tool_market_use_nonce_v1(p_hotkey TEXT,p_signature_hash TEXT,p_signed_at BIGINT)
RETURNS BOOLEAN LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $nonce770$
BEGIN
 INSERT INTO public.tool_market_nonces(hotkey,signature_hash,signed_at) VALUES(p_hotkey,p_signature_hash,p_signed_at)
 ON CONFLICT DO NOTHING;
 RETURN FOUND;
END;
$nonce770$;

CREATE FUNCTION public.tool_market_accept_terms_v1(p_role TEXT,p_hotkey TEXT,p_version TEXT,p_terms_sha256 TEXT,
 p_host TEXT,p_signed_at BIGINT,p_body_sha256 TEXT,p_signature TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $terms770$
BEGIN
 INSERT INTO public.tool_market_terms(role,hotkey,terms_sha256,terms_version,host,signed_at,body_sha256,signature)
 VALUES(p_role,p_hotkey,p_terms_sha256,p_version,p_host,p_signed_at,p_body_sha256,p_signature)
 ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status',CASE WHEN FOUND THEN 'accepted' ELSE 'exists' END);
END;
$terms770$;

CREATE FUNCTION public.tool_market_open_tab_v1(p_tab_id UUID,p_hotkey TEXT,p_token_hash TEXT,p_budget BIGINT,
 p_seconds INTEGER,p_terms_sha256 TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $open_tab770$
DECLARE balance JSONB; expires TIMESTAMPTZ;
BEGIN
 PERFORM public.tool_market__lock_v1('buyer',p_hotkey);
 IF NOT EXISTS(SELECT 1 FROM public.tool_market_terms WHERE role='buyer' AND hotkey=p_hotkey
   AND terms_sha256=p_terms_sha256) THEN
  RETURN jsonb_build_object('status','terms_required');
 END IF;
 balance:=public.tool_market_balance_v1(p_hotkey);
 IF NOT (balance->>'available')::BOOLEAN OR (balance->>'balance_microusd')::BIGINT<p_budget THEN
  RETURN jsonb_build_object('status','balance_low','balance',balance);
 END IF;
 expires:=clock_timestamp()+make_interval(secs=>p_seconds);
 INSERT INTO public.tool_market_tabs(tab_id,hotkey,token_hash,budget_microusd,expires_at)
 VALUES(p_tab_id,p_hotkey,p_token_hash,p_budget,expires);
 RETURN jsonb_build_object('status','opened','tab_id',p_tab_id,'budget_microusd',p_budget,'expires_at',expires);
END;
$open_tab770$;

CREATE FUNCTION public.tool_market_tab_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $tab770$
 SELECT jsonb_build_object('tab_id',t.tab_id,'hotkey',t.hotkey,'budget_microusd',t.budget_microusd,
  'committed_microusd',public.tool_market__committed_v1(t.tab_id),'expires_at',t.expires_at,
  'open',public.tool_market__tab_open_v1(t),
  'calls',(SELECT count(*) FROM public.tool_market_calls c WHERE c.tab_id=t.tab_id))
 FROM public.tool_market_tabs t WHERE t.token_hash=p_token_hash;
$tab770$;

CREATE FUNCTION public.tool_market_close_tab_v1(p_token_hash TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $close770$
DECLARE tab public.tool_market_tabs;
BEGIN
 SELECT * INTO tab FROM public.tool_market_tabs WHERE token_hash=p_token_hash;
 IF tab.tab_id IS NULL THEN RETURN NULL; END IF;
 -- The buyer lock, as reserve and open_tab: no call is reserved on a tab
 -- whose close another tab's budget already counts on.
 PERFORM public.tool_market__lock_v1('buyer',tab.hotkey);
 INSERT INTO public.tool_market_tab_closures(tab_id) VALUES(tab.tab_id) ON CONFLICT DO NOTHING;
 RETURN public.tool_market_tab_v1(p_token_hash);
END;
$close770$;

-- ---------------------------------------------------------------------------
-- Miners: keys, health, asks
-- ---------------------------------------------------------------------------

-- One active listing per key: the same vendor key sealed twice, by one miner
-- or two, is refused while the first is listed. For security, a hotkey
-- registers at most one key per tool every 30 days after a key that served a
-- call, revoked keys included. A key that never served one does not count.
CREATE FUNCTION public.tool_market_register_key_v1(p_key_id UUID,p_tool TEXT,p_hotkey TEXT,p_envelope TEXT,
 p_fingerprint TEXT,p_daily_cap BIGINT,p_status TEXT,p_balance NUMERIC,p_host TEXT,p_signed_at BIGINT,
 p_signed_body TEXT,p_signature TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $register770$
DECLARE next_at TIMESTAMPTZ;
BEGIN
 IF p_status NOT IN ('valid','unchecked') THEN
  RAISE EXCEPTION 'only a valid or unchecked key is registered' USING ERRCODE='22023';
 END IF;
 PERFORM public.tool_market__lock_v1('miner_tool',p_hotkey||':'||p_tool);
 SELECT max(k.registered_at)+interval '30 days' INTO next_at FROM public.tool_market_keys k
  WHERE k.miner_hotkey=p_hotkey AND k.tool=p_tool AND EXISTS(
   SELECT 1 FROM public.tool_market_calls c JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id
    WHERE c.key_id=k.key_id AND s.status='settled');
 IF next_at>clock_timestamp() THEN
  RETURN jsonb_build_object('status','too_soon','next_at',next_at);
 END IF;
 PERFORM public.tool_market__lock_v1('fingerprint',p_fingerprint);
 IF EXISTS(SELECT 1 FROM public.tool_market_keys k WHERE k.key_fingerprint=p_fingerprint
   AND public.tool_market__key_active_v1(k.key_id)) THEN
  RETURN jsonb_build_object('status','duplicate');
 END IF;
 INSERT INTO public.tool_market_keys(key_id,tool,miner_hotkey,envelope,key_fingerprint,daily_cap_microusd,
  host,signed_at,signed_body,signature)
 VALUES(p_key_id,p_tool,p_hotkey,p_envelope,p_fingerprint,p_daily_cap,p_host,p_signed_at,p_signed_body,p_signature);
 INSERT INTO public.tool_market_key_events(key_id,status,balance) VALUES(p_key_id,p_status,p_balance);
 RETURN jsonb_build_object('status','registered','key_id',p_key_id);
END;
$register770$;

CREATE FUNCTION public.tool_market_key_v1(p_key_id UUID)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $key770$
 SELECT jsonb_build_object('key_id',k.key_id,'tool',k.tool,'miner_hotkey',k.miner_hotkey,'envelope',k.envelope,
  'key_fingerprint',k.key_fingerprint,'active',public.tool_market__key_active_v1(k.key_id))
 FROM public.tool_market_keys k WHERE k.key_id=p_key_id;
$key770$;

CREATE FUNCTION public.tool_market_revoke_key_v1(p_key_id UUID,p_hotkey TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $revoke770$
BEGIN
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);  -- after any reserve in flight
 IF NOT EXISTS(SELECT 1 FROM public.tool_market_keys WHERE key_id=p_key_id AND miner_hotkey=p_hotkey) THEN
  RETURN jsonb_build_object('status','not_found');
 END IF;
 INSERT INTO public.tool_market_key_revocations(key_id,reason) VALUES(p_key_id,p_reason) ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status','revoked');
END;
$revoke770$;

CREATE FUNCTION public.tool_market_record_key_event_v1(p_key_id UUID,p_status TEXT,p_reason TEXT,p_balance NUMERIC)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $event770$
BEGIN
 INSERT INTO public.tool_market_key_events(key_id,status,reason,balance) VALUES(p_key_id,p_status,p_reason,p_balance);
 RETURN jsonb_build_object('status','recorded');
END;
$event770$;

CREATE FUNCTION public.tool_market_post_ask_v1(p_key_id UUID,p_hotkey TEXT,p_ask BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $ask770$
DECLARE seq BIGINT;
BEGIN
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);  -- after any reserve in flight
 IF NOT EXISTS(SELECT 1 FROM public.tool_market_keys WHERE key_id=p_key_id AND miner_hotkey=p_hotkey)
  OR NOT public.tool_market__key_active_v1(p_key_id) THEN
  RETURN jsonb_build_object('status','not_found');
 END IF;
 INSERT INTO public.tool_market_asks(key_id,ask_microusd) VALUES(p_key_id,p_ask) RETURNING ask_sequence INTO seq;
 RETURN jsonb_build_object('status',CASE WHEN p_ask IS NULL THEN 'withdrawn' ELSE 'posted' END,'ask_sequence',seq);
END;
$ask770$;

-- The tool's tradeable supply. A key trades while it is listed, its miner
-- accepted the current terms, its latest ask is a price, its latest event
-- is valid or unchecked, and its last 24 h are under its daily cap.
-- ``needs_check``: its vendor has a free check (it was last ``valid``) and
-- that check is older than ``p_check_max_age_seconds``.
CREATE FUNCTION public.tool_market_book_v1(p_tool TEXT,p_terms_sha256 TEXT,p_check_max_age_seconds INTEGER)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $book770$
 WITH keyed AS (
  SELECT k.key_id,k.miner_hotkey,k.envelope,k.key_fingerprint,k.daily_cap_microusd,
   (SELECT a FROM public.tool_market_asks a WHERE a.key_id=k.key_id ORDER BY a.ask_sequence DESC LIMIT 1) ask,
   (SELECT e FROM public.tool_market_key_events e WHERE e.key_id=k.key_id ORDER BY e.event_id DESC LIMIT 1) event
  FROM public.tool_market_keys k
  WHERE k.tool=p_tool AND public.tool_market__key_active_v1(k.key_id)
   AND EXISTS(SELECT 1 FROM public.tool_market_terms m WHERE m.role='miner' AND m.hotkey=k.miner_hotkey
    AND m.terms_sha256=p_terms_sha256))
 SELECT coalesce(jsonb_agg(jsonb_build_object('key_id',key_id,'miner_hotkey',miner_hotkey,'envelope',envelope,
   'key_fingerprint',key_fingerprint,'ask_microusd',(ask).ask_microusd,'ask_sequence',(ask).ask_sequence,
   'needs_check',(event).status='valid'
     AND (event).recorded_at<clock_timestamp()-make_interval(secs=>p_check_max_age_seconds))
   ORDER BY (ask).ask_microusd,(ask).ask_sequence),'[]'::JSONB)
 FROM keyed
 WHERE (ask).ask_microusd IS NOT NULL AND (event).status IN ('valid','unchecked')
  AND public.tool_market__key_day_v1(key_id)<daily_cap_microusd;
$book770$;

-- ---------------------------------------------------------------------------
-- Calls
-- ---------------------------------------------------------------------------

-- Hold one call's reserve against its tab, at the key's current ask.
CREATE FUNCTION public.tool_market_reserve_call_v1(p_call_id UUID,p_token_hash TEXT,p_tool TEXT,p_operation TEXT,
 p_key_id UUID,p_ask BIGINT,p_reserve BIGINT,p_owed_cap BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $reserve770$
DECLARE tab public.tool_market_tabs; k public.tool_market_keys; current_ask BIGINT;
BEGIN
 SELECT * INTO tab FROM public.tool_market_tabs WHERE token_hash=p_token_hash;
 IF tab.tab_id IS NULL THEN RETURN jsonb_build_object('status','tab_closed'); END IF;
 -- Lock order: buyer, then key. The buyer lock orders this reserve against
 -- a close of this tab and the balance check of a new tab, so a tab is
 -- open, checked under the lock, whenever a call is added to it.
 PERFORM public.tool_market__lock_v1('buyer',tab.hotkey);
 IF NOT public.tool_market__tab_open_v1(tab) THEN
  RETURN jsonb_build_object('status','tab_closed');
 END IF;
 IF public.tool_market__committed_v1(tab.tab_id)+p_reserve>tab.budget_microusd THEN
  RETURN jsonb_build_object('status','tab_budget');
 END IF;
 PERFORM public.tool_market__lock_v1('key',p_key_id::TEXT);
 SELECT * INTO k FROM public.tool_market_keys WHERE key_id=p_key_id AND tool=p_tool;
 SELECT ask_microusd INTO current_ask FROM public.tool_market_asks WHERE key_id=p_key_id
  ORDER BY ask_sequence DESC LIMIT 1;
 IF k.key_id IS NULL OR NOT public.tool_market__key_active_v1(p_key_id) OR current_ask IS DISTINCT FROM p_ask THEN
  RETURN jsonb_build_object('status','ask_changed');
 END IF;
 IF k.miner_hotkey=tab.hotkey THEN  -- a miner's own tab never buys from their own key
  RETURN jsonb_build_object('status','own_key');
 END IF;
 IF public.tool_market__key_day_v1(p_key_id)+p_owed_cap>k.daily_cap_microusd THEN
  RETURN jsonb_build_object('status','key_cap');
 END IF;
 INSERT INTO public.tool_market_calls(call_id,tab_id,tool,operation,key_id,miner_hotkey,ask_microusd,reserve_microusd,
  owed_cap_microusd)
 VALUES(p_call_id,tab.tab_id,p_tool,p_operation,p_key_id,k.miner_hotkey,p_ask,p_reserve,p_owed_cap);
 RETURN jsonb_build_object('status','reserved');
END;
$reserve770$;

-- Settle once, within the open window. A settlement may not charge more
-- than the call reserved.
CREATE FUNCTION public.tool_market_settle_call_v1(p_call_id UUID,p_status TEXT,p_http_status INTEGER,p_units BIGINT,
 p_owed BIGINT,p_charge BIGINT,p_epoch BIGINT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $settle770$
DECLARE c public.tool_market_calls;
BEGIN
 SELECT * INTO c FROM public.tool_market_calls WHERE call_id=p_call_id;
 IF c.call_id IS NULL THEN RETURN jsonb_build_object('status','not_found'); END IF;
 IF c.opened_at<=clock_timestamp()-make_interval(secs=>public.tool_market__open_seconds_v1()) THEN
  RETURN jsonb_build_object('status','expired');
 END IF;
 IF p_charge>c.reserve_microusd THEN
  RAISE EXCEPTION 'a call cannot charge more than it reserved' USING ERRCODE='22023';
 END IF;
 INSERT INTO public.tool_market_call_settlements(call_id,status,http_status,units,owed_microusd,charge_microusd,epoch)
 VALUES(p_call_id,p_status,p_http_status,p_units,p_owed,p_charge,p_epoch) ON CONFLICT DO NOTHING;
 RETURN jsonb_build_object('status',CASE WHEN FOUND THEN 'settled' ELSE 'exists' END);
END;
$settle770$;

-- A miner's keys, their state and what each has earned.
CREATE FUNCTION public.tool_market_statement_v1(p_hotkey TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $statement770$
 SELECT jsonb_build_object('keys',coalesce(jsonb_agg(jsonb_build_object(
   'key_id',k.key_id,'tool',k.tool,'key_fingerprint',k.key_fingerprint,
   'active',public.tool_market__key_active_v1(k.key_id),'daily_cap_microusd',k.daily_cap_microusd,
   'ask_microusd',(SELECT a.ask_microusd FROM public.tool_market_asks a WHERE a.key_id=k.key_id
     ORDER BY a.ask_sequence DESC LIMIT 1),
   'health',(SELECT jsonb_build_object('status',e.status,'reason',e.reason,'at',e.recorded_at)
     FROM public.tool_market_key_events e WHERE e.key_id=k.key_id ORDER BY e.event_id DESC LIMIT 1),
   'calls',(SELECT count(*) FROM public.tool_market_calls c WHERE c.key_id=k.key_id),
   'uncertain_calls',(SELECT count(*) FROM public.tool_market_calls c
     JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id WHERE c.key_id=k.key_id AND s.status='uncertain'),
   'owed_microusd',(SELECT coalesce(sum(s.owed_microusd),0) FROM public.tool_market_calls c
     JOIN public.tool_market_call_settlements s ON s.call_id=c.call_id WHERE c.key_id=k.key_id))
   ORDER BY k.sequence),'[]'::JSONB))
 FROM public.tool_market_keys k WHERE k.miner_hotkey=p_hotkey;
$statement770$;

-- Every tool_market function: owner, no PUBLIC execute; the service may
-- execute all but the double-underscore helpers.
DO $functions770$
DECLARE f REGPROCEDURE; name TEXT;
BEGIN
 FOR f,name IN SELECT p.oid::REGPROCEDURE,p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
  WHERE n.nspname='public' AND p.proname LIKE 'tool\_market%\_v1' LOOP
  EXECUTE format('ALTER FUNCTION %s OWNER TO lab_arena_owner',f);
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC,anon,authenticated,service_role,lab_arena_service',f);
  IF name NOT LIKE 'tool\_market\_\_%' THEN
   EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO lab_arena_service',f);
  END IF;
 END LOOP;
END;
$functions770$;

DO $restore770$
BEGIN
 IF NOT(SELECT had_create FROM pg_temp.tool_market770_acl)THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;END IF;
 IF(SELECT nspacl FROM pg_namespace WHERE nspname='public')IS DISTINCT FROM(SELECT nspacl FROM pg_temp.tool_market770_acl)THEN
  RAISE EXCEPTION 'tool_market770 schema ACL differs';END IF;
END;
$restore770$;
NOTIFY pgrst,'reload schema';
COMMIT;
